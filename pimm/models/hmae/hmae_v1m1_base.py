"""
Hierarchical Masked Autoencoder (HMAE) v1m1

A hierarchical point cloud masked autoencoder that:
1. Groups points into patches at the coarsest encoder level
2. Masks a fraction of patches
3. Encodes only visible patches through PTv3 encoder
4. Decodes masked patches via cross-attention on encoder features
5. Reconstructs points within each masked patch using chamfer loss

Author: Based on Sonata framework by Xiaoyang Wu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from pimm.models.utils.structure import Point
from pimm.models.builder import MODELS, build_model
from pimm.models.modules import PointModel
from pimm.models.losses.chamfer import ChamferLoss


class PositionalEncodingMLP(nn.Module):
    """MLP-based positional encoding for 3D centroids."""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 256, hidden_channels: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, centroids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            centroids: (N, 3) patch centroid coordinates
        Returns:
            pos_embed: (N, out_channels) positional embeddings
        """
        return self.mlp(centroids)


class DecoderSelfAttention(nn.Module):
    """Self-attention layer for mask tokens."""
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int = 2048, # default max sequence length for flash attention
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C) packed mask tokens
            cu_seqlens: (B+1,) cumulative sequence lengths
            max_seqlen: maximum sequence length
        """
        H = self.num_heads
        C = self.channels
        
        qkv = self.qkv(x)
        
        if flash_attn is not None:
            qkv = qkv.reshape(-1, 3, H, C // H)
            out = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.to(torch.bfloat16),
                cu_seqlens,
                max_seqlen=max_seqlen,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            )
            out = out.reshape(-1, C).to(x.dtype)
        else:
            # fallback to standard attention with masking
            N = x.shape[0]
            q, k, v = qkv.reshape(-1, 3, H, C // H).permute(1, 0, 2, 3).unbind(0)
            
            # vectorized block-diagonal mask creation (no Python loops)
            idx = torch.arange(N, device=x.device)
            batch_ids = torch.bucketize(idx, cu_seqlens[1:], right=False)
            attn_mask = torch.where(
                batch_ids.unsqueeze(0) == batch_ids.unsqueeze(1),
                torch.tensor(0.0, device=x.device, dtype=x.dtype),
                torch.tensor(float('-inf'), device=x.device, dtype=x.dtype),
            )
            
            # compute attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn + attn_mask.unsqueeze(1)  # (N, H, N)
            attn = F.softmax(attn, dim=-1)
            
            out = (attn @ v).reshape(-1, C)
        
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class DecoderCrossAttention(nn.Module):
    """Cross-attention layer: mask tokens attend to encoder features."""
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.k_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.v_proj = nn.Linear(channels, channels, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        q_cu_seqlens: torch.Tensor,
        kv_cu_seqlens: torch.Tensor,
        max_seqlen_q: int = 2048, # default max sequence length for flash attention
        max_seqlen_kv: int = 2048, # default max sequence length for flash attention
    ) -> torch.Tensor:
        """
        Args:
            query: (N_q, C) mask tokens
            key, value: (N_kv, C) encoder features
            q_cu_seqlens: (B+1,) cumulative lengths for queries
            kv_cu_seqlens: (B+1,) cumulative lengths for keys/values
        """
        H = self.num_heads
        C = self.channels
        
        q = self.q_proj(query).reshape(-1, H, C // H)
        k = self.k_proj(key).reshape(-1, H, C // H)
        v = self.v_proj(value).reshape(-1, H, C // H)
        
        if flash_attn is not None:
            out = flash_attn.flash_attn_varlen_func(
                q.to(torch.bfloat16),
                k.to(torch.bfloat16),
                v.to(torch.bfloat16),
                q_cu_seqlens,
                kv_cu_seqlens,
                max_seqlen_q,
                max_seqlen_kv,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            )
            out = out.reshape(-1, C).to(query.dtype)
        else:
            # fallback to standard attention with masking
            N_q = query.shape[0]
            N_kv = key.shape[0]
            
            # vectorized cross-attention mask (no Python loops)
            q_idx = torch.arange(N_q, device=query.device)
            kv_idx = torch.arange(N_kv, device=query.device)
            q_batch = torch.bucketize(q_idx, q_cu_seqlens[1:], right=False)
            kv_batch = torch.bucketize(kv_idx, kv_cu_seqlens[1:], right=False)
            attn_mask = torch.where(
                q_batch.unsqueeze(1) == kv_batch.unsqueeze(0),
                torch.tensor(0.0, device=query.device, dtype=query.dtype),
                torch.tensor(float('-inf'), device=query.device, dtype=query.dtype),
            )
            
            attn = torch.einsum('qhd,khd->qhk', q, k) * self.scale
            attn = attn + attn_mask.unsqueeze(1)
            attn = F.softmax(attn, dim=-1)
            
            if self.training and self.attn_drop > 0:
                attn = F.dropout(attn, p=self.attn_drop)
            
            out = torch.einsum('qhk,khd->qhd', attn, v)
            out = out.reshape(-1, C)
        
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class DecoderMLP(nn.Module):
    """MLP block for decoder."""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = None,
        out_channels: int = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels * 4
        
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DecoderBlock(nn.Module):
    """
    Single decoder block: self-attention -> cross-attention -> MLP
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        # self.norm1 = nn.LayerNorm(channels)
        # self.self_attn = DecoderSelfAttention(
        #     channels, num_heads, qkv_bias, attn_drop, proj_drop
        # )
        
        self.norm1 = nn.LayerNorm(channels)
        self.cross_attn = DecoderCrossAttention(
            channels, num_heads, qkv_bias, attn_drop, proj_drop
        )
        
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = DecoderMLP(
            channels,
            hidden_channels=int(channels * mlp_ratio),
            drop=proj_drop,
        )
        
        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        encoder_feat: torch.Tensor,
        q_cu_seqlens: torch.Tensor,
        kv_cu_seqlens: torch.Tensor,
        max_seqlen_q: int = 2048,
        max_seqlen_kv: int = 2048,
    ) -> torch.Tensor:
        # self-attention
        # residual = x
        # x = self.norm1(x)
        # x = self.self_attn(x, q_cu_seqlens, max_seqlen_q)
        # if self.training and self.drop_path > 0:
        #     x = x * (torch.rand(1, device=x.device) > self.drop_path).float()
        # x = residual + x
        
        # cross-attention
        residual = x
        x = self.norm1(x)
        x = self.cross_attn(
            x, encoder_feat, encoder_feat,
            q_cu_seqlens, kv_cu_seqlens,
            max_seqlen_q, max_seqlen_kv
        )
        x = residual + self.drop_path(x)
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.drop_path(x)
        
        return x


class HMAEDecoder(nn.Module):
    """
    HMAE Decoder: transforms mask tokens into point predictions.
    """
    
    def __init__(
        self,
        channels: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        mlp_ratio: float = 4.0,
        points_per_patch: int = 128,
        output_dim: int = 4,  # x, y, z, energy
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.points_per_patch = points_per_patch
        self.output_dim = output_dim
        
        # learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, channels))
        trunc_normal_(self.mask_token, std=0.02)
        
        # positional encoding for centroids
        self.pos_enc = PositionalEncodingMLP(
            in_channels=3,
            out_channels=channels,
            hidden_channels=channels // 2,
        )
        
        # decoder blocks
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.blocks = nn.ModuleList([
            DecoderBlock(
                channels=channels,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path_rates[i],
            )
            for i in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(channels)
        
        # output projection: predict K points with (x, y, z, energy)
        self.output_proj = nn.Linear(channels, points_per_patch * output_dim)
    
    def forward(
        self,
        input: Point,
    ) -> torch.Tensor:
        """
        Args:
            encoder_feat: (N_enc, C) encoder features (packed)
            encoder_offset: (B,) cumulative offsets for encoder features
            encoder_coord: (N_enc, 3) encoder coordinates (packed, final stage)
            masked_centroids: (N_mask, 3) centroids of masked patches (packed)
            masked_offset: (B,) cumulative offsets for masked patches
        
        Returns:
            predictions: (N_mask, K, output_dim) predicted points per patch
        """
        # encoder_feat: torch.Tensor,
        # encoder_offset: torch.Tensor,
        # encoder_coord: torch.Tensor,
        # masked_centroids: torch.Tensor,
        # masked_offset: torch.Tensor,
        masked_centroids = input.masked_centroids

        N_mask = masked_centroids.shape[0]
        device = masked_centroids.device
        
        # create mask tokens with positional encoding
        mask_tokens = self.mask_token.expand(N_mask, -1)
        mask_pos_embed = self.pos_enc(masked_centroids)
        x = mask_tokens
        
        # add positional encoding to encoder keys/values (backbone outputs)
        kv_pos_embed = self.pos_enc(input.coord)
        encoder_feat = input.feat + kv_pos_embed.to(input.feat.dtype)
        
        # compute cu_seqlens for flash attention
        q_cu_seqlens = torch.cat([input.masked_offset.new_zeros(1), input.masked_offset]).int()
        kv_cu_seqlens = torch.cat([input.offset.new_zeros(1), input.offset]).int()
        
        # compute max sequence lengths directly from cu_seqlens (faster than offset2bincount)
        
        # apply decoder blocks
        for block in self.blocks:
            x = block(
                x + mask_pos_embed, encoder_feat,
                q_cu_seqlens, kv_cu_seqlens,
                # max_seqlen_q, max_seqlen_kv
            )
        
        x = self.norm(x)
        
        # project to point predictions
        pred = self.output_proj(x)  # (N_mask, K * output_dim)
        pred = pred.reshape(N_mask, self.points_per_patch, self.output_dim) # (B, K, output_dim)
        
        return pred


@MODELS.register_module("HMAE-v1m1")
class HMAE(PointModel):
    """
    Hierarchical Masked Autoencoder for Point Clouds.
    """
    
    def __init__(
        self,
        backbone,
        decoder_channels: int = 256,
        decoder_num_heads: int = 8,
        decoder_num_layers: int = 3,
        decoder_mlp_ratio: float = 4.0,
        points_per_patch: int = 128,
        patch_size: float = 0.016,
        mask_ratio: float = 0.6,
        coord_loss_weight: float = 1.0,
        energy_loss_weight: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.points_per_patch = points_per_patch
        self.grid_size = None
        
        # build encoder (PTv3 in enc_mode)
        self.backbone = build_model(backbone)
        
        # get encoder output channels (from last encoder stage)
        enc_channels = backbone['enc_channels']
        encoder_out_channels = enc_channels[-1]
        
        # projection from encoder to decoder dimension if needed
        if encoder_out_channels != decoder_channels:
            self.encoder_proj = nn.Linear(encoder_out_channels, decoder_channels)
        else:
            self.encoder_proj = nn.Identity()
        
        # decoder
        self.decoder = HMAEDecoder(
            channels=decoder_channels,
            num_heads=decoder_num_heads,
            num_layers=decoder_num_layers,
            mlp_ratio=decoder_mlp_ratio,
            points_per_patch=points_per_patch,
            output_dim=4,  # x, y, z, energy (relative coords)
            drop_path=drop_path,
        )
        
        # loss: chamfer for coords, MSE on matched points for energy
        self.chamfer_loss = ChamferLoss(
            coord_weight=coord_loss_weight,
            feat_weight=energy_loss_weight,
            norm=2,
        )
            
    def forward(self, data_dict):
        """Forward with preprocessed data from transform."""
        visible_coord = data_dict["visible_coord"]
        visible_feat = data_dict['visible_energy']
        visible_offset = data_dict["visible_offset"]
        
        masked_centroids = data_dict["masked_centroids"]
        masked_offset = data_dict["masked_offset"]
        
        target_coords = data_dict["target_coords"]  # (total_points, 3) flattened
        target_feat = data_dict['target_energy']  # (total_points, 1) flattened
        target_offset = data_dict["target_offset"]  # (n_patches,) cumulative offsets
        
        # grid_size may be batched into a tensor - extract scalar
        if self.grid_size is None:
            grid_size = data_dict.get("grid_size", 0.001)
            if isinstance(grid_size, torch.Tensor):
                grid_size = grid_size[0] if grid_size.numel() > 1 else grid_size
            self.grid_size = grid_size
        
        if masked_centroids.shape[0] == 0:
            return {"loss": torch.tensor(0.0, device=visible_coord.device, requires_grad=True)}
        
        # encode visible points
        visible_origin_coord = data_dict.get("visible_origin_coord", visible_coord)
        
        encoder_input = Point(
            feat=visible_feat,
            coord=visible_coord,
            origin_coord=visible_origin_coord,
            offset=visible_offset,
            grid_size=self.grid_size,
        )
        
        encoder_output = self.backbone(encoder_input)
        encoder_feat = self.encoder_proj(encoder_output.feat)
        encoder_offset = encoder_output.offset
        encoder_coord = encoder_output.coord
        
        # decode
        decoder_input = Point(
            feat=encoder_feat,
            offset=encoder_offset,
            coord=encoder_coord,
            masked_centroids=masked_centroids,
            masked_offset=masked_offset,
        )
        predictions = self.decoder(decoder_input)
        
        # predictions shape: (N_mask, points_per_patch, output_dim)
        # flatten to (N_mask * points_per_patch, output_dim) for offset batching
        pred_coord = predictions[..., :3]  # (N_mask, points_per_patch, 3)
        pred_energy = predictions[..., 3:4]  # (N_mask, points_per_patch, 1)
        
        # flatten predictions
        pred_coord_flat = pred_coord.reshape(-1, 3)  # (N_mask * points_per_patch, 3)
        pred_energy_flat = pred_energy.reshape(-1, 1)  # (N_mask * points_per_patch, 1)
        
        # ensure target_offset is a torch tensor on the correct device
        # target_offset from HMAECollate is cumulative offsets with shape (B,): [n1, n1+n2, ...]
        if not isinstance(target_offset, torch.Tensor):
            target_offset = torch.from_numpy(target_offset).long()
        target_offset = target_offset.to(pred_coord.device).long()
        
        # only take the first k points per event, where k is the actual number of target points
        # target_offset is cumulative with shape (B,): [n1, n1+n2, ...] for B samples
        # masked_offset is cumulative patch counts: [p1, p1+p2, ...] for B samples (shape: B, doesn't start with 0)
        # we need to slice predictions to match the actual number of points per batch sample
        if masked_offset.shape[0] > 0:
            # create cumulative offsets for full predictions (all points_per_patch points)
            # prepend 0 to masked_offset to make it cumulative with starting 0
            masked_offset_cum = torch.cat([
                torch.zeros(1, device=masked_offset.device, dtype=masked_offset.dtype),
                masked_offset
            ])
            full_pred_offset = masked_offset_cum * self.points_per_patch
            
            # prepend 0 to target_offset for easier indexing
            target_offset_cum = torch.cat([
                torch.zeros(1, device=target_offset.device, dtype=target_offset.dtype),
                target_offset
            ])
            
            # slice predictions to only include first k points per batch sample
            batch_size = target_offset.shape[0]
            total_points = target_offset[-1].item() if batch_size > 0 else 0
            pred_coord_flat_new = pred_coord_flat.new_zeros((total_points, 3))
            pred_energy_flat_new = pred_energy_flat.new_zeros((total_points, 1))

            for i in range(batch_size):
                # get range for this batch sample in full predictions
                start_idx = full_pred_offset[i]
                end_idx = full_pred_offset[i + 1]

                # get actual number of target points for this batch sample
                target_start = target_offset_cum[i]
                target_end = target_offset_cum[i + 1]
                num_actual_points = target_end - target_start

                pred_coord_flat_new[target_start:target_end] = pred_coord_flat[start_idx:end_idx][:num_actual_points]
                pred_energy_flat_new[target_start:target_end] = pred_energy_flat[start_idx:end_idx][:num_actual_points]

            pred_coord_flat = pred_coord_flat_new
            pred_energy_flat = pred_energy_flat_new
            
            # pred_offset should match target_offset exactly since we slice to match target points per batch
            # use target_offset directly to ensure shapes match (already shape B,)
            pred_offset = target_offset.clone()
        else:
            # fallback: use target_offset if no masked patches
            num_actual_points = target_offset[-1].item() if target_offset.shape[0] > 0 else 0
            pred_coord_flat = pred_coord_flat[:num_actual_points]
            pred_energy_flat = pred_energy_flat[:num_actual_points]
            # pred_offset should match target_offset since we're using all target points
            pred_offset = target_offset.clone()
        
        # ensure target_feat has correct shape
        if target_feat is not None:
            if target_feat.ndim == 1:
                target_feat = target_feat[:, None]

        
        loss = self.chamfer_loss(
            pred_coord=pred_coord_flat.float(),
            target_coord=target_coords.float(),
            pred_feat=pred_energy_flat.float(),
            target_feat=target_feat.float(),
            pred_offset=pred_offset.int(),
            target_offset=target_offset.int(),
        )
        return loss