
import argparse
import os
from collections import defaultdict
from itertools import chain
from random import sample

from torch.utils.data import ConcatDataset, Subset
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm
import numpy as np
from model.RelationNet_event import LocalUncertaintyAwareGraphAttentionLite
from tool.warmup_lr import warmup_decay_cosine
from tool.kinetics_metric import save_submission_in_seconds_multi_thresh, eval_by_frame_metric_from_seconds_map, \
    summarize_global_avg_over_thresholds
from dataset.kinetics.dataset import build, collate_fn
import os


def build_center_reg_labels_from_relative_targets(targets_list):
    B = len(targets_list)
    device = (targets_list[0]['boundaries'].device
              if torch.is_tensor(targets_list[0]['boundaries'])
              else torch.device('cpu'))

    cls_label = torch.zeros(B, dtype=torch.float32, device=device)
    center_gt = torch.full((B,), 0.5, dtype=torch.float32, device=device)
    has_pos = torch.zeros(B, dtype=torch.bool, device=device)
    vid_ids = []

    for i, t in enumerate(targets_list):
        vid_tensor = t.get('video_id', None)
        vid = int(vid_tensor.item()) if torch.is_tensor(vid_tensor) else (vid_tensor if vid_tensor is not None else i)
        vid_ids.append(str(vid))

        y = t['boundaries']
        if y.numel() > 0:
            cls_label[i] = 1.0
            idx = torch.argmin(torch.abs(y - 0.5))
            center_gt[i] = y[idx]
            has_pos[i] = True
        else:
            cls_label[i] = 0.0

    return cls_label, center_gt, has_pos, vid_ids


@torch.no_grad()
def map_center_to_abs_frames(center_pred, base, window_size, interval, locations=None):
    if locations is None:
        win_span = window_size * interval
        abs_frames = base.float() + center_pred.float() * float(win_span)
        return abs_frames
    else:
        B, T, _ = locations.shape
        t = center_pred.clamp(0, 1) * (T - 1)
        abs_frames = []
        for b in range(B):
            ys = locations[b, :, 0]
            tb = t[b]
            lo = torch.clamp(tb.floor().long(), 0, T - 1)
            hi = torch.clamp(lo + 1, 0, T - 1)
            wb = tb - lo.float()
            abs_b = ys[lo] * (1 - wb) + ys[hi] * wb
            abs_frames.append(abs_b)
        return torch.stack(abs_frames, dim=0)


def temporal_consistency_loss_margin(center_pred, cls_label, margin=0.05, neighborhood_func=None):
    pos_mask = (cls_label > 0.5).float()
    num_pos = pos_mask.sum().clamp(min=1.0)
    device = center_pred.device

    if not pos_mask.any():
        return torch.tensor(0.0, device=device, dtype=center_pred.dtype)

    # 向量化：(P, B) 广播，避免 Python for 循环
    pred_pos = center_pred[pos_mask > 0.5]           # (P,)
    diff = torch.abs(pred_pos.unsqueeze(1) - center_pred.unsqueeze(0))  # (P, B)
    loss_tcl = torch.clamp(margin - diff, min=0.0).mean(dim=1).sum() / num_pos
    return loss_tcl


def temporal_consistency_loss_kl(center_pred, center_pred_next, cls_label, margin=0.05):
    neg_mask = (cls_label < 0.5).float()
    eps = 1e-7
    p_t = torch.clamp(center_pred, eps, 1 - eps)
    p_next = torch.clamp(center_pred_next, eps, 1 - eps)
    kl_div = (p_t * torch.log(p_t / p_next) +
              (1 - p_t) * torch.log((1 - p_t) / (1 - p_next)))
    loss_tcl = (kl_div * neg_mask).sum() / neg_mask.sum().clamp(min=1.0)
    return loss_tcl


def event_boundary_detection_loss(
        score_logit, center_pred, cls_label,
        center_pred_next=None, use_tcl_margin=True,
        margin=0.05, lambda_tcl=1.0
):
    loss_det = F.binary_cross_entropy_with_logits(score_logit, cls_label.float())

    if use_tcl_margin:
        loss_tcl = temporal_consistency_loss_margin(center_pred, cls_label, margin=margin)
    else:
        if center_pred_next is None:
            raise ValueError("center_pred_next 必须提供以计算 KL 版本的 TCL")
        loss_tcl = temporal_consistency_loss_kl(center_pred, center_pred_next, cls_label, margin=margin)

    loss_total = loss_det + lambda_tcl * loss_tcl
    loss_dict = {
        'det': loss_det.detach().item(),
        'tcl': loss_tcl.detach().item(),
        'total': loss_total.detach().item()
    }
    return loss_total, loss_dict


def train_epoch(
        trainload, model, opti, lr_sh, gpu=0,
        pos_weight=None, lambda_tcl=1.0, use_tcl=False,
        margin=0.05, use_tcl_margin=True
):
    model.train().cuda(gpu)
    progress = tqdm(trainload, desc="Train(Event Boundary Detection)")

    total_loss = 0.0
    total_loss_det = 0.0
    total_loss_tcl = 0.0
    num_batches = 0

    for sample in progress:
        locations, features, targets_list, num_frames, base, coherence = sample
        features = features.cuda(gpu, non_blocking=True)

        cls_label, center_gt, has_pos, _ = build_center_reg_labels_from_relative_targets(targets_list)
        cls_label = cls_label.cuda(gpu, non_blocking=True)
        center_gt = center_gt.cuda(gpu, non_blocking=True)
        has_pos = has_pos.cuda(gpu, non_blocking=True)

        opti.zero_grad(set_to_none=True)
        score_logit, center_pred = model(features)

        if use_tcl:
            loss, loss_dict = event_boundary_detection_loss(
                score_logit=score_logit, center_pred=center_pred,
                cls_label=cls_label, center_pred_next=None,
                use_tcl_margin=use_tcl_margin, margin=margin, lambda_tcl=lambda_tcl
            )
            total_loss_det += loss_dict['det']
            total_loss_tcl += loss_dict['tcl']
        else:
            loss = F.binary_cross_entropy_with_logits(score_logit, cls_label.float())
            total_loss_det += loss.detach().item()

        loss.backward()
        opti.step()
        if lr_sh is not None:
            lr_sh.step()

        total_loss += loss.detach().item()
        num_batches += 1

        if use_tcl:
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                det=f"{loss_dict['det']:.4f}",
                tcl=f"{loss_dict['tcl']:.4f}"
            )
        else:
            progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    avg_loss_det = total_loss_det / max(num_batches, 1)
    avg_loss_tcl = total_loss_tcl / max(num_batches, 1) if use_tcl else 0.0
    return {'total': avg_loss, 'det': avg_loss_det, 'tcl': avg_loss_tcl}


@torch.no_grad()
def test_epoch(testload, model, gpu=0):
    model.eval().cuda(gpu)
    results_dict = defaultdict(list)

    for batch in tqdm(testload, desc="Evaluate(Event Dataset)"):
        locations, features, targets_list, num_frames, base, coherence = batch
        B, L, _ = features.shape
        features = features.cuda(gpu, non_blocking=True)

        score_logit, center_pred = model(features)
        probs = torch.sigmoid(score_logit)

        base_b = base.view(-1).to(features.device).float()
        loc = locations.to(features.device).float()

        t = center_pred.clamp(0, 1) * (L - 1)
        lo = t.floor().long().clamp(0, L - 1)
        hi = (lo + 1).clamp(0, L - 1)
        w = t - lo.float()
        idx = torch.arange(B, device=features.device)
        rel_abs = loc[idx, lo, 0] * (1 - w) + loc[idx, hi, 0] * w
        abs_frames = (base_b + rel_abs).detach().cpu()
        probs_cpu = probs.detach().cpu()

        for i, tdict in enumerate(targets_list):
            vid_tensor = tdict.get("video_id", None)
            vid = str(
                int(vid_tensor.item()) if torch.is_tensor(vid_tensor)
                else (vid_tensor if vid_tensor is not None else i)
            )
            results_dict[vid].append({
                "boundaries": abs_frames[i:i + 1],
                "scores": probs_cpu[i:i + 1],
            })
    return dict(results_dict)


def evaluate_multi_thresholds(
        results, anno_root_path,
        thresholds=np.linspace(0.05, 0.50, 10),
        rel_dist_list=(0.10,),
        step_frames=1,
        out_dir='multif-pred_outputs',
        write_pkls=False,
        debug=True
):
    seconds_map_by_thresh = save_submission_in_seconds_multi_thresh(
        results, anno_root_path, thresholds,
        step_frames=step_frames, out_dir=out_dir,
        write_pkls=write_pkls, debug=debug,
    )
    eval_results = eval_by_frame_metric_from_seconds_map(
        anno_root_path, seconds_map_by_thresh,
        downsample=1, rel_dist_list=rel_dist_list,
        filter_low_consis=True, consis_th=0.3, debug=debug
    )
    return eval_results


def run_experiment(args, window_size, size, exp_tag, gpu_id):
    """
    单次实验：给定 window_size 和 size，训练并评估，保存 checkpoint。
    exp_tag: checkpoint 文件名标识，如 'window=9' 或 'size=17'
    """
    print(f"\n{'='*60}")
    print(f"  实验开始: {exp_tag}  (window_size={window_size}, size={size})")
    print(f"{'='*60}\n")

    # 重新构建数据集（window_size 影响 dataset 的滑窗策略）
    args.window_size = size
    train_dataset = build(split='train', args=args)
    val_dataset = build(split='val', args=args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    model = LocalUncertaintyAwareGraphAttentionLite(
        dim_in=2048,
        num_heads=8,
        window_size=window_size,   # 本次实验的 window_size
        sim_temperature=0.07,
        neighbor_temp=0.5,
        uncertainty_mode="variance",
        use_relative_pos_bias=True,
        norm="ln",
        size=size                  # 本次实验的 size
    ).to(args.device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4,
    )
    lr_sh = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_decay_cosine(len(train_loader), len(train_loader) * 19)
    )

    EPOCHS = 1
    USE_TCL = False
    LAMBDA_TCL = 1.0
    MARGIN = 0.1
    USE_TCL_MARGIN = False

    thresholds = [0.1,0.2,0.3,0.4, 0.50]
    rel_d_list = (0.05, 0.10, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)

    for ep in range(EPOCHS):
        loss_stats = train_epoch(
            trainload=train_loader,
            model=model,
            opti=optimizer,
            lr_sh=lr_sh,
            gpu=gpu_id,
            lambda_tcl=LAMBDA_TCL,
            use_tcl=USE_TCL,
            margin=MARGIN,
            use_tcl_margin=USE_TCL_MARGIN
        )

        print(f"\n[{exp_tag}] Epoch {ep + 1}/{EPOCHS} | "
              f"Loss(total)={loss_stats['total']:.4f}, "
              f"Loss(det)={loss_stats['det']:.4f}, "
              f"Loss(tcl)={loss_stats['tcl']:.4f}")

        # 保存 checkpoint，以实验标识命名
        ckpt_name = f"event_{exp_tag}_ep{ep}.pth"
        ckpt_path = os.path.join(args.output_dir, ckpt_name)
        torch.save({'model': model.state_dict()}, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

        # 验证
        results_dict = test_epoch(val_loader, model, gpu=gpu_id)
        eval_res = evaluate_multi_thresholds(
            results_dict, anno_root_path=args.annotation_path,
            thresholds=thresholds,
            rel_dist_list=rel_d_list,
            step_frames=1,
            out_dir=f'multif-pred_outputs/{exp_tag}',
            write_pkls=False,
            debug=True,
        )


    print(f"\n  [{exp_tag}] 实验完成。\n")


def main():
    args = argparse.Namespace()
    args.feature_path = "/data/shared_dataset/Kinetics/features"
    args.score_path = "/data/shared_dataset/Kinetics/data"
    args.annotation_path = "/data/shared_dataset/Kinetics/data"
    args.interval = 1
    args.device = 'cuda:3'
    args.output_dir = '/home/tianxiaoxuan/data/mamba/task1/checkpoint_event_new'
    os.makedirs(args.output_dir, exist_ok=True)

    gpu_id = int(args.device.split(':')[-1])

    # ─────────────────────────────────────────────────────────────
    # 第一组实验：window_size ∈ {5, 7, 9, 11}，size 固定为 21
    # ─────────────────────────────────────────────────────────────
    # FIXED_SIZE = 21
    # for ws in [5,7,9,11]:
    #     run_experiment(
    #         args=args,
    #         window_size=ws,
    #         size=FIXED_SIZE,
    #         exp_tag=f"window={ws}",
    #         gpu_id=gpu_id
    #     )

    # # ─────────────────────────────────────────────────────────────
    # # 第二组实验：window_size 固定为 9，size ∈ {13, 17, 25}
    # # ─────────────────────────────────────────────────────────────
    # FIXED_WINDOW = 9
    # for sz in [13, 17, 25]:
    #     run_experiment(
    #         args=args,
    #         window_size=FIXED_WINDOW,
    #         size=sz,
    #         exp_tag=f"size={sz}",
    #         gpu_id=gpu_id
    #     )


    run_experiment(
        args=args,
        window_size=9,
        size=21,
        exp_tag=f"size={21}",
        gpu_id=gpu_id
    )

if __name__ == '__main__':
    main()
