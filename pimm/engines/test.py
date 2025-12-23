"""
Tester for semantic and panoptic segmentation

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import numpy as np
from collections import OrderedDict, defaultdict
import torch
import torch.distributed as dist
import pointops
from sklearn.metrics import adjusted_rand_score

from .defaults import create_ddp_model
import pimm.utils.comm as comm
from pimm.datasets import build_dataset, collate_fn
from pimm.models import build_model
from pimm.utils.logger import get_root_logger
from pimm.utils.registry import Registry
from pimm.utils.misc import (
    intersection_and_union_gpu,
    make_dirs,
)
from pimm.models.utils.misc import offset2bincount


TESTERS = Registry("testers")


class TesterBase:
    def __init__(self, cfg, model=None, test_loader=None, verbose=False) -> None:
        torch.multiprocessing.set_sharing_strategy("file_system")
        self.logger = get_root_logger(
            log_file=os.path.join(cfg.save_path, "test.log"),
            file_mode="a" if cfg.resume else "w",
        )
        self.logger.info("=> Loading config ...")
        self.cfg = cfg
        self.verbose = verbose
        if self.verbose:
            self.logger.info(f"Save path: {cfg.save_path}")
            self.logger.info(f"Config:\n{cfg.pretty_text}")
        if model is None:
            self.logger.info("=> Building model ...")
            self.model = self.build_model()
        else:
            self.model = model
        if test_loader is None:
            self.logger.info("=> Building test dataset & dataloader ...")
            self.test_loader = self.build_test_loader()
        else:
            self.test_loader = test_loader

    def build_model(self):
        model = build_model(self.cfg.model)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.logger.info(f"Num params: {n_parameters}")
        model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )
        if os.path.isfile(self.cfg.weight):
            self.logger.info(f"Loading weight at: {self.cfg.weight}")
            checkpoint = torch.load(self.cfg.weight)
            weight = OrderedDict()
            for key, value in checkpoint["state_dict"].items():
                if key.startswith("module."):
                    if comm.get_world_size() == 1:
                        key = key[7:]  # module.xxx.xxx -> xxx.xxx
                else:
                    if comm.get_world_size() > 1:
                        key = "module." + key  # xxx.xxx -> module.xxx.xxx
                weight[key] = value
            model.load_state_dict(weight, strict=True)
            self.logger.info(
                "=> Loaded weight '{}' (epoch {})".format(
                    self.cfg.weight, checkpoint.get("epoch", "unknown")
                )
            )
        else:
            raise RuntimeError("=> No checkpoint found at '{}'".format(self.cfg.weight))
        return model

    def build_test_loader(self):
        test_dataset = build_dataset(self.cfg.data.test)
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.batch_size_test_per_gpu,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader

    def test(self):
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class SemSegTester(TesterBase):
    """Semantic segmentation tester following SemSegEvaluator pattern."""

    def __init__(
        self,
        cfg,
        model=None,
        test_loader=None,
        verbose=False,
        ignore_index=-1,
        macro_ignore_class_ids=None,
    ):
        super().__init__(cfg, model, test_loader, verbose)
        self.ignore_index = ignore_index
        self.macro_ignore_class_ids = tuple(sorted(set(macro_ignore_class_ids or [])))

    def test(self):
        self.logger.info(">>>>>>>>>>>>>>>> Start Semantic Segmentation Test >>>>>>>>>>>>>>>>")
        self.model.eval()

        all_preds = []
        all_segments = []

        for i, input_dict in enumerate(self.test_loader):
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.model(input_dict)

            if "seg_logits" in output_dict:
                output = output_dict["seg_logits"]
            elif "sem_logits" in output_dict:
                output = output_dict["sem_logits"]
            else:
                raise KeyError(
                    "No semantic logits found in model output (expected 'seg_logits' or 'sem_logits')."
                )

            pred = output.max(1)[1]
            segment = input_dict["segment"]

            if "origin_coord" in input_dict.keys():
                idx, _ = pointops.knn_query(
                    1,
                    input_dict["coord"].float(),
                    input_dict["offset"].int(),
                    input_dict["origin_coord"].float(),
                    input_dict["origin_offset"].int(),
                )
                pred = pred[idx.flatten().long()]
                segment = input_dict["origin_segment"]

            segment = segment.squeeze(-1)

            all_preds.append(pred.cpu())
            all_segments.append(segment.cpu())

            intersection, union, target = intersection_and_union_gpu(
                pred,
                segment,
                self.cfg.data.num_classes,
                self.cfg.data.ignore_index,
            )
            if comm.get_world_size() > 1:
                dist.all_reduce(intersection)
                dist.all_reduce(union)
                dist.all_reduce(target)

            info = "Test: [{iter}/{max_iter}]".format(
                iter=i + 1, max_iter=len(self.test_loader)
            )
            if "origin_coord" in input_dict.keys():
                info = "Interp. " + info
            self.logger.info(info)

        if comm.get_world_size() > 1:
            all_preds_gathered = comm.gather(all_preds, dst=0)
            all_segments_gathered = comm.gather(all_segments, dst=0)
            if comm.get_rank() == 0:
                all_preds = [p for preds in all_preds_gathered for p in preds]
                all_segments = [s for segments in all_segments_gathered for s in segments]
            else:
                self.model.train()
                return

        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_segments = torch.cat(all_segments, dim=0).numpy()

        num_classes = self.cfg.data.num_classes
        precision_class = np.zeros(num_classes)
        recall_class = np.zeros(num_classes)
        f1_class = np.zeros(num_classes)

        for i in range(num_classes):
            pred_i = all_preds == i
            gt_i = all_segments == i
            if gt_i.sum() > 0 or pred_i.sum() > 0:
                tp = np.logical_and(pred_i, gt_i).sum()
                fp = np.logical_and(pred_i, np.logical_not(gt_i)).sum()
                fn = np.logical_and(np.logical_not(pred_i), gt_i).sum()

                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)

                precision_class[i] = precision
                recall_class[i] = recall
                f1_class[i] = f1

        # compute intersection/union/target for IoU and accuracy
        intersection = np.zeros(num_classes)
        union = np.zeros(num_classes)
        target = np.zeros(num_classes)

        for i in range(num_classes):
            pred_i = all_preds == i
            gt_i = all_segments == i
            intersection[i] = np.logical_and(pred_i, gt_i).sum()
            union[i] = np.logical_or(pred_i, gt_i).sum()
            target[i] = gt_i.sum()

        iou_class = intersection / (union + 1e-10)
        acc_class = intersection / (target + 1e-10)

        macro_mask = np.ones(num_classes, dtype=bool)
        for idx in self.macro_ignore_class_ids:
            if 0 <= idx < num_classes:
                macro_mask[idx] = False

        precision_valid = precision_class[macro_mask]
        recall_valid = recall_class[macro_mask]
        f1_valid = f1_class[macro_mask]
        iou_valid = iou_class[macro_mask]
        acc_valid = acc_class[macro_mask]

        if precision_valid.size == 0:
            precision_valid = precision_class
        if recall_valid.size == 0:
            recall_valid = recall_class
        if f1_valid.size == 0:
            f1_valid = f1_class
        if iou_valid.size == 0:
            iou_valid = iou_class
        if acc_valid.size == 0:
            acc_valid = acc_class

        m_precision = np.mean(precision_valid)
        m_recall = np.mean(recall_valid)
        m_f1 = np.mean(f1_valid)
        m_iou = np.mean(iou_valid)
        m_acc = np.mean(acc_valid)
        all_acc = sum(intersection) / (sum(target) + 1e-10)

        self.logger.info(
            "Test result: mIoU/mAcc/allAcc/mPrec/mRec/mF1 {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.".format(
                m_iou, m_acc, all_acc, m_precision, m_recall, m_f1
            )
        )

        table_header = "| Class ID | Class Name | IoU | Accuracy | Precision | Recall | F1 |"
        table_separator = (
            "|"
            + "-" * 10
            + "|"
            + "-" * 12
            + "|"
            + "-" * 8
            + "|"
            + "-" * 10
            + "|"
            + "-" * 11
            + "|"
            + "-" * 8
            + "|"
            + "-" * 6
            + "|"
        )

        self.logger.info("Per-class metrics:")
        self.logger.info(table_header)
        self.logger.info(table_separator)

        if not macro_mask.all():
            self.logger.info("* indicates class ignored in macro metrics")

        for i in range(self.cfg.data.num_classes):
            ignored_marker = "*" if not macro_mask[i] else ""
            self.logger.info(
                "| {idx:8d} | {name:10s} | {iou:.4f} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} |".format(
                    idx=i,
                    name=(self.cfg.data.names[i] + ignored_marker),
                    iou=iou_class[i],
                    accuracy=acc_class[i],
                    precision=precision_class[i],
                    recall=recall_class[i],
                    f1=f1_class[i],
                )
            )

        self.logger.info("<<<<<<<<<<<<<<<<< End Semantic Segmentation Test <<<<<<<<<<<<<<<<<")
        self.model.train()

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)


@TESTERS.register_module()
class InstanceSegTester(TesterBase):
    """Panoptic/Instance segmentation tester following InstanceSegmentationEvaluator pattern."""

    def __init__(
        self,
        cfg,
        model=None,
        test_loader=None,
        verbose=False,
        stuff_threshold=0.5,
        mask_threshold=0.5,
        class_names=None,
        stuff_classes=None,
        iou_thresh=0.5,
        require_class_for_match=False,
    ):
        super().__init__(cfg, model, test_loader, verbose)
        self.stuff_threshold = float(stuff_threshold)
        self.mask_threshold = float(mask_threshold)
        self.iou_thresh = float(iou_thresh)
        self.require_class_for_match = bool(require_class_for_match)
        self.class_names = tuple(class_names or [])
        self.stuff_classes = tuple(sorted(stuff_classes or ()))

    def _unwrap(self, obj):
        if isinstance(obj, torch.nn.parallel.DistributedDataParallel):
            return obj.module
        return obj

    def test(self):
        self.logger.info(
            ">>>>>>>>>>>>>> Start Panoptic/Instance Segmentation Test >>>>>>>>>>>>>>>>"
        )
        self.model.eval()

        class_names = (
            self.class_names
            if len(self.class_names)
            else tuple(getattr(self.cfg.data, "names", []))
        )
        if not class_names:
            class_names = tuple(range(self.cfg.data.num_classes))

        all_stats = []
        ari_scores = []
        momentum_stats = []

        for input_dict in self.test_loader:
            assert (
                len(input_dict["offset"]) == 1
            ), "InstanceSegTester requires bs=1"

            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.cuda(non_blocking=True)

            with torch.no_grad():
                output_dict = self.model(input_dict, return_point=True)

            point = output_dict.get("point")
            if point is None:
                self.logger.warning("InstanceSegTester: missing point data")
                continue

            gt_instance = point.instance
            if gt_instance is None:
                self.logger.warning("InstanceSegTester: missing instance labels")
                continue

            gt_segment = getattr(point, "segment", None)
            if gt_segment is None:
                self.logger.warning("InstanceSegTester: missing PID labels")
                continue

            pred_masks_list = output_dict.get("pred_masks")
            pred_logits_list = output_dict.get("pred_logits")
            pred_momentum_list = output_dict.get("pred_momentum")
            if not pred_masks_list or pred_logits_list is None:
                self.logger.warning("InstanceSegTester: missing predictions")
                continue

            model = self._unwrap(self.model)

            stuff_probs = (
                point.outputs.get("stuff_probs")
                if hasattr(point, "outputs")
                else None
            )
            point_counts = offset2bincount(point.offset)

            post_input = {
                "pred_masks": pred_masks_list,
                "pred_logits": pred_logits_list,
                "stuff_probs": stuff_probs,
                "point_counts": point_counts,
                "pred_momentum": pred_momentum_list,
            }

            results = model.postprocess(
                post_input,
                stuff_threshold=self.stuff_threshold,
                mask_threshold=self.mask_threshold,
            )

            pred_instance_labels = results["instance_labels"].cpu()
            pred_pid_labels = results["class_labels"].cpu()

            pred_instance_momentum = None
            if "pred_momentum" in results:
                pred_instance_momentum = results["pred_momentum"].cpu()

            gt_inst = gt_instance.squeeze(-1).cpu().numpy().astype(np.int64)
            gt_pid = gt_segment.squeeze(-1).cpu().numpy().astype(np.int64)
            pr_inst = pred_instance_labels.numpy().astype(np.int64)
            pr_pid = pred_pid_labels.numpy().astype(np.int64)

            stats = self._eval_instances(
                gt_inst,
                pr_inst,
                gt_pid,
                pr_pid,
                class_names=class_names,
                iou_thresh=self.iou_thresh,
                require_class_for_match=self.require_class_for_match,
            )
            all_stats.append(stats)

            momentum_gt = input_dict.get("momentum")
            if pred_instance_momentum is not None and momentum_gt is not None:
                counts = offset2bincount(point.offset)
                counts_list = counts.cpu().tolist()

                criterion = model.criteria.criteria[0].criterion

                mom_stats = self._eval_momentum(
                    stats["matches"],
                    pred_instance_momentum,
                    momentum_gt,
                    gt_instance,
                    counts_list,
                    criterion,
                    num_classes=self.cfg.data.num_classes,
                    pred_instance_labels=pr_inst,
                )
                if mom_stats is not None:
                    momentum_stats.append(mom_stats)

            led_mask = pr_inst != -1
            if led_mask.any():
                ari_scores.append(
                    adjusted_rand_score(gt_inst[led_mask], pr_inst[led_mask])
                )
            else:
                ari_scores.append(float("nan"))

        if comm.get_world_size() > 1:
            gathered = comm.gather((all_stats, ari_scores, momentum_stats), dst=0)
            if comm.get_rank() == 0:
                merged_stats = []
                merged_ari = []
                merged_momentum = []
                for stats_chunk, ari_chunk, momentum_chunk in gathered:
                    merged_stats.extend(stats_chunk)
                    merged_ari.extend(ari_chunk)
                    merged_momentum.extend(momentum_chunk)
                all_stats = merged_stats
                ari_scores = merged_ari
                momentum_stats = merged_momentum
            else:
                self.model.train()
                return

        if not all_stats:
            self.logger.warning("InstanceSegTester: no stats computed")
            self.model.train()
            return

        aggregated = self._aggregate_instance_results(
            all_stats, require_class_for_match=self.require_class_for_match
        )

        det = aggregated["detection"]
        cls = aggregated["classification_on_matched"]
        det_prec = det["precision"]
        det_rec = det["recall"]
        det_f1 = det["f1"]
        det_iou = det["mean_matched_iou"]
        total_gt = int(det.get("num_gt", 0))
        total_pred = int(det.get("num_pred", 0))
        total_matched = int(det.get("num_matched", 0))
        fp_det = max(total_pred - total_matched, 0)
        fn_det = max(total_gt - total_matched, 0)
        fp_per_gt = (fp_det / total_gt) if total_gt > 0 else 0.0
        fn_per_gt = (fn_det / total_gt) if total_gt > 0 else 0.0

        support = cls["support"]
        if np.any(support > 0):
            precision_macro = float(np.mean(cls["precision"][support > 0]))
            recall_macro = float(np.mean(cls["recall"][support > 0]))
            f1_macro = float(np.mean(cls["f1"][support > 0]))
        else:
            precision_macro = recall_macro = f1_macro = 0.0

        ari_clean = np.asarray(ari_scores, dtype=float)
        ari_clean = ari_clean[~np.isnan(ari_clean)]
        ari_mean = float(np.mean(ari_clean)) if ari_clean.size else float("nan")

        self.logger.info(
            "Detection P={:.3f} R={:.3f} F1={:.3f} IoU={:.3f}".format(
                det_prec, det_rec, det_f1, det_iou
            )
        )
        self.logger.info(
            "Counts GT={} Pred={} TP={} FP={} FN={} (FP/GT={:.3f} FN/GT={:.3f})".format(
                total_gt, total_pred, total_matched, fp_det, fn_det, fp_per_gt, fn_per_gt
            )
        )
        self.logger.info(
            "Classification macro P={:.3f} R={:.3f} F1={:.3f}".format(
                precision_macro, recall_macro, f1_macro
            )
        )
        if not np.isnan(ari_mean):
            self.logger.info("ARI mean={:.3f}".format(ari_mean))
        else:
            self.logger.info("ARI mean=n/a")

        if self.require_class_for_match and "pq" in aggregated:
            pq = aggregated["pq"]
            self.logger.info(
                "Panoptic Quality: PQ={:.3f} RQ={:.3f} SQ={:.3f}".format(
                    pq["PQ"], pq["RQ"], pq["SQ"]
                )
            )

        if momentum_stats:
            momentum_aggregated = self._aggregate_momentum_results(momentum_stats, class_names)
            self._log_momentum_metrics(momentum_aggregated, class_names)

        self.logger.info(
            "<<<<<<<<<<<<<<<<< End Panoptic/Instance Segmentation Test <<<<<<<<<<<<<<<<<"
        )
        self.model.train()

    def _eval_instances(
        self,
        gt_inst,
        pr_inst,
        gt_pid,
        pr_pid,
        class_names,
        iou_thresh=0.5,
        require_class_for_match=False,
    ):
        """Evaluate instance segmentation metrics for a single event."""
        K = len(class_names)
        N = gt_inst.shape[0]
        assert pr_inst.shape[0] == N

        valid_mask = (gt_inst >= 0) & (pr_inst >= 0)

        pr_ids, pr_sizes = np.unique(pr_inst[pr_inst >= 0], return_counts=True)
        gt_ids, gt_sizes = np.unique(gt_inst[gt_inst >= 0], return_counts=True)
        pr_size = {int(i): int(c) for i, c in zip(pr_ids, pr_sizes)}
        gt_size = {int(i): int(c) for i, c in zip(gt_ids, gt_sizes)}

        if valid_mask.any():
            pairs = np.stack([pr_inst[valid_mask], gt_inst[valid_mask]], axis=1)
            uniq_pairs, inter_counts = np.unique(pairs, axis=0, return_counts=True)
            inter_map = {
                (int(p), int(g)): int(c) for (p, g), c in zip(uniq_pairs, inter_counts)
            }
        else:
            inter_map = {}

        def build_inst_pid_map(inst_ids_present, inst_ids_per_point, pid_array):
            inst_ids_present = list(map(int, inst_ids_present))
            m = int(np.max(inst_ids_present)) + 1 if inst_ids_present else 0

            if pid_array.ndim == 1 and pid_array.shape[0] >= m and m > 0:
                return {i: int(pid_array[i]) for i in inst_ids_present if 0 <= int(pid_array[i]) < K}

            pid_map = {}
            if len(inst_ids_present) == 0:
                return pid_map
            use = (inst_ids_per_point >= 0) & (pid_array >= 0) & (pid_array < K)
            inst_vals = inst_ids_per_point[use].astype(np.int64)
            pid_vals = pid_array[use].astype(np.int64)
            ip = np.stack([inst_vals, pid_vals], axis=1)
            uniq_ip, cnts = np.unique(ip, axis=0, return_counts=True)
            order = np.lexsort((-cnts, uniq_ip[:, 0]))
            uniq_ip_sorted = uniq_ip[order]
            cnts_sorted = cnts[order]
            _, first_idx = np.unique(uniq_ip_sorted[:, 0], return_index=True)
            for idx in first_idx:
                inst_i = int(uniq_ip_sorted[idx, 0])
                pid_i = int(uniq_ip_sorted[idx, 1])
                if 0 <= pid_i < K:
                    pid_map[inst_i] = pid_i
            return pid_map

        gt_pid_map = build_inst_pid_map(gt_ids, gt_inst, gt_pid)
        pr_pid_map = build_inst_pid_map(pr_ids, pr_inst, pr_pid)

        def class_ok(p_id, g_id):
            if not require_class_for_match:
                return True
            return pr_pid_map.get(p_id, -999) == gt_pid_map.get(g_id, -998)

        cand = []
        for (p, g), inter in inter_map.items():
            if p not in pr_size or g not in gt_size:
                continue
            if not class_ok(p, g):
                continue
            union = pr_size[p] + gt_size[g] - inter
            if union <= 0:
                continue
            iou = inter / union
            cand.append((iou, p, g, inter))

        cand.sort(reverse=True, key=lambda t: t[0])
        used_p, used_g = set(), set()
        matches = []
        for iou, p, g, inter in cand:
            if iou < iou_thresh:
                break
            if p in used_p or g in used_g:
                continue
            matches.append((p, g, iou, inter))
            used_p.add(p)
            used_g.add(g)

        num_gt = len(gt_ids)
        num_pred = len(pr_ids)
        num_matched = len(matches)
        fp = num_pred - num_matched
        fn = num_gt - num_matched

        det_prec = num_matched / (num_matched + fp) if (num_matched + fp) else 0.0
        det_rec = num_matched / (num_matched + fn) if (num_matched + fn) else 0.0
        det_f1 = (
            (2 * det_prec * det_rec / (det_prec + det_rec)) if (det_prec + det_rec) else 0.0
        )
        mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0

        confusion = np.zeros((K, K), dtype=int)
        out_matches = []
        iou_sum_for_pq = 0.0
        tp_for_pq = 0
        for p, g, iou, inter in matches:
            pred_cls = pr_pid_map.get(p, -1)
            gt_cls = gt_pid_map.get(g, -1)
            if 0 <= gt_cls < K and 0 <= pred_cls < K:
                confusion[gt_cls, pred_cls] += 1
            out_matches.append(
                {
                    "pred_id": p,
                    "gt_id": g,
                    "iou": float(iou),
                    "pred_cls": int(pred_cls),
                    "gt_cls": int(gt_cls),
                    "intersection": int(inter),
                    "pred_size": pr_size[p],
                    "gt_size": gt_size[g],
                }
            )
            if require_class_for_match and pred_cls == gt_cls and 0 <= pred_cls < K:
                tp_for_pq += 1
                iou_sum_for_pq += iou

        support = np.array([confusion[i, :].sum() for i in range(K)], dtype=int)
        precision = np.zeros(K)
        recall = np.zeros(K)
        f1 = np.zeros(K)
        for i in range(K):
            tp = confusion[i, i]
            fp_c = confusion[:, i].sum() - tp
            fn_c = confusion[i, :].sum() - tp
            pr = tp / (tp + fp_c) if (tp + fp_c) else 0.0
            rc = tp / (tp + fn_c) if (tp + fn_c) else 0.0
            precision[i], recall[i] = pr, rc
            f1[i] = (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0

        out = {
            "detection": {
                "precision": det_prec,
                "recall": det_rec,
                "f1": det_f1,
                "num_gt": num_gt,
                "num_pred": num_pred,
                "num_matched": num_matched,
                "mean_matched_iou": mean_iou,
                "iou_thresh": iou_thresh,
                "matching": "class-aware" if require_class_for_match else "unlabeled",
            },
            "classification_on_matched": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion": confusion,
                "support": support,
                "class_names": list(class_names),
            },
            "matches": out_matches,
        }

        if require_class_for_match:
            sq = (iou_sum_for_pq / tp_for_pq) if tp_for_pq else 0.0
            rq_den = tp_for_pq + 0.5 * fp + 0.5 * fn
            rq = (tp_for_pq / rq_den) if rq_den > 0 else 0.0
            out["pq"] = {"PQ": rq * sq, "RQ": rq, "SQ": sq}

        return out

    def _aggregate_instance_results(self, stats_list, require_class_for_match=False):
        """Aggregate instance results across events."""
        if len(stats_list) == 0:
            raise ValueError("stats_list is empty")

        cls0 = stats_list[0]["classification_on_matched"]
        class_names = list(cls0["class_names"])
        K = len(class_names)

        total_gt = total_pred = total_matched = 0
        iou_sum_all_matches = 0.0
        iou_sum_pq = 0.0
        tp_pq = 0

        pooled_conf = np.zeros((K, K), dtype=int)
        pooled_support = np.zeros(K, dtype=int)

        for s in stats_list:
            det = s["detection"]
            total_gt += det["num_gt"]
            total_pred += det["num_pred"]
            total_matched += det["num_matched"]

            for m in s["matches"]:
                iou_sum_all_matches += float(m["iou"])
                if 0 <= m["gt_cls"] < K and m["gt_cls"] == m["pred_cls"]:
                    tp_pq += 1
                    iou_sum_pq += float(m["iou"])

            cls = s["classification_on_matched"]
            pooled_conf += np.asarray(cls["confusion"], dtype=int)
            pooled_support += np.asarray(cls["support"], dtype=int)

        tp_det = total_matched
        fp_det = total_pred - total_matched
        fn_det = total_gt - total_matched

        det_prec = tp_det / (tp_det + fp_det) if (tp_det + fp_det) else 0.0
        det_rec = tp_det / (tp_det + fn_det) if (tp_det + fn_det) else 0.0
        det_f1 = (
            (2 * det_prec * det_rec / (det_prec + det_rec)) if (det_prec + det_rec) else 0.0
        )
        mean_iou = (iou_sum_all_matches / tp_det) if tp_det else 0.0

        tp = np.diag(pooled_conf)
        fp = pooled_conf.sum(axis=0) - tp
        fn = pooled_conf.sum(axis=1) - tp

        precision = np.zeros(K)
        recall = np.zeros(K)
        f1 = np.zeros(K)
        for i in range(K):
            pr = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) else 0.0
            rc = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) else 0.0
            precision[i], recall[i] = pr, rc
            f1[i] = (2 * pr * rc / (pr + rc)) if (pr + rc) else 0.0

        res_global = {
            "detection": {
                "precision": det_prec,
                "recall": det_rec,
                "f1": det_f1,
                "num_gt": int(total_gt),
                "num_pred": int(total_pred),
                "num_matched": int(total_matched),
                "mean_matched_iou": float(mean_iou),
                "iou_thresh": stats_list[0]["detection"]["iou_thresh"],
                "matching": stats_list[0]["detection"]["matching"],
            },
            "classification_on_matched": {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "confusion": pooled_conf,
                "support": pooled_support,
                "class_names": class_names,
            },
            "matches": [],
        }

        if require_class_for_match:
            rq_den = tp_pq + 0.5 * fp_det + 0.5 * fn_det
            rq = (tp_pq / rq_den) if rq_den > 0 else 0.0
            sq = (iou_sum_pq / tp_pq) if tp_pq > 0 else 0.0
            res_global["pq"] = {"PQ": rq * sq, "RQ": rq, "SQ": sq}

        return res_global

    def _eval_momentum(
        self,
        matches,
        pred_instance_momentum,
        momentum_gt,
        gt_instance,
        counts,
        criterion,
        num_classes,
        pred_instance_labels,
    ):
        """Evaluate momentum regression per class for matched instances."""
        try:
            b = 0
            mom_gt_b = criterion._get_batch_tensor(
                momentum_gt, b, counts, torch.device("cpu"), None
            )

            inst_b = gt_instance.squeeze(-1)
            if inst_b.dim() == 0:
                inst_b = inst_b.unsqueeze(0)
            if inst_b.dim() == 2 and inst_b.shape[1] == 1:
                inst_b = inst_b.squeeze(1)
            inst_b = inst_b.cpu()

            unique_gt_ids = torch.unique(inst_b)
            unique_gt_ids = unique_gt_ids[unique_gt_ids >= 0]
            gt_mom_map = {}
            for gt_id in unique_gt_ids:
                mask = inst_b == gt_id
                if mask.any():
                    gt_mom_map[gt_id.item()] = mom_gt_b[mask].float().mean().item()

            if isinstance(pred_instance_labels, np.ndarray):
                pred_inst_t = torch.from_numpy(pred_instance_labels)
            else:
                pred_inst_t = pred_instance_labels

            sorted_ids, sort_idx = torch.sort(pred_inst_t)
            unique_ids_sorted, counts_sorted = torch.unique_consecutive(sorted_ids, return_counts=True)

            cumsum_counts = torch.cat([torch.tensor([0]), counts_sorted.cumsum(0)[:-1]])
            first_indices = sort_idx[cumsum_counts]
            mom_values = pred_instance_momentum[first_indices]

            pr_mom_map = {
                uid.item(): val.item()
                for uid, val in zip(unique_ids_sorted, mom_values)
                if uid.item() >= 0
            }

            matched_preds = []
            for m in matches:
                pid = m["pred_id"]
                gid = m["gt_id"]
                pred_cls = m["pred_cls"]

                if pid in pr_mom_map and gid in gt_mom_map:
                    matched_preds.append(
                        {"pred": pr_mom_map[pid], "gt": gt_mom_map[gid], "cls": pred_cls}
                    )

            if not matched_preds:
                return None

            all_p = np.array([x["pred"] for x in matched_preds])
            all_g = np.array([x["gt"] for x in matched_preds])
            all_c = np.array([x["cls"] for x in matched_preds])
            all_e = all_p - all_g

            class_stats = {}
            for cls_idx in range(num_classes):
                cls_mask = all_c == cls_idx
                if not np.any(cls_mask):
                    continue

                cls_errors = all_e[cls_mask]
                cls_mae = float(np.mean(np.abs(cls_errors)))
                cls_rmse = float(np.sqrt(np.mean(cls_errors ** 2)))
                cls_count = int(np.sum(cls_mask))

                class_stats[cls_idx] = {
                    "mae": cls_mae,
                    "rmse": cls_rmse,
                    "count": cls_count,
                }

            overall_mae = float(np.mean(np.abs(all_e)))
            overall_rmse = float(np.sqrt(np.mean(all_e ** 2)))

            return {
                "per_class": class_stats,
                "overall": {
                    "mae": overall_mae,
                    "rmse": overall_rmse,
                    "count": len(all_e),
                },
            }

        except Exception as e:
            self.logger.warning(f"Error evaluating momentum: {e}")
            import traceback

            traceback.print_exc()
            return None

    def _aggregate_momentum_results(self, momentum_stats_list, class_names):
        """Aggregate momentum evaluation results across batches."""
        num_classes = len(class_names)

        class_mae = {i: [] for i in range(num_classes)}
        class_rmse = {i: [] for i in range(num_classes)}
        class_count = {i: 0 for i in range(num_classes)}

        overall_mae = []
        overall_rmse = []
        overall_count = 0

        for stats in momentum_stats_list:
            if stats is None:
                continue

            for cls_idx, cls_stats in stats["per_class"].items():
                if cls_idx < num_classes:
                    class_mae[cls_idx].append(cls_stats["mae"])
                    class_rmse[cls_idx].append(cls_stats["rmse"])
                    class_count[cls_idx] += cls_stats["count"]

            overall_mae.append(stats["overall"]["mae"])
            overall_rmse.append(stats["overall"]["rmse"])
            overall_count += stats["overall"]["count"]

        aggregated_per_class = {}
        for cls_idx in range(num_classes):
            if len(class_mae[cls_idx]) > 0:
                aggregated_per_class[cls_idx] = {
                    "mae": float(np.mean(class_mae[cls_idx])),
                    "rmse": float(np.mean(class_rmse[cls_idx])),
                    "count": class_count[cls_idx],
                }

        aggregated_overall = {
            "mae": float(np.mean(overall_mae)) if overall_mae else 0.0,
            "rmse": float(np.mean(overall_rmse)) if overall_rmse else 0.0,
            "count": overall_count,
        }

        return {
            "per_class": aggregated_per_class,
            "overall": aggregated_overall,
        }

    def _log_momentum_metrics(self, momentum_aggregated, class_names):
        """Log momentum regression metrics."""
        self.logger.info("Momentum Regression Metrics:")

        overall = momentum_aggregated["overall"]
        self.logger.info(
            "Overall: MAE={:.4f} RMSE={:.4f} Count={}".format(
                overall["mae"], overall["rmse"], overall["count"]
            )
        )

        per_class = momentum_aggregated["per_class"]
        if per_class:
            self.logger.info("Per-class metrics:")
            for cls_idx in sorted(per_class.keys()):
                cls_name = (
                    class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
                )
                cls_stats = per_class[cls_idx]
                self.logger.info(
                    "  {}: MAE={:.4f} RMSE={:.4f} Count={}".format(
                        cls_name, cls_stats["mae"], cls_stats["rmse"], cls_stats["count"]
                    )
                )

    @staticmethod
    def collate_fn(batch):
        return collate_fn(batch)
