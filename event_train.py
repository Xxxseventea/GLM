import argparse
import os
from collections import defaultdict

import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm
from model.backbone.RelationNet_ import LocalUncertaintyAwareGraphAttentionLite
from tool.warmup_lr import warmup_decay_cosine
from tool.metric.kinetics_metric import save_submission_in_seconds_multi_thresh,eval_by_frame_metric_from_seconds_map,summarize_global_avg_over_thresholds
from dataset.kinetics.dataset import build, collate_fn  # 假设代码保存在某个文件
import os
import pickle
import numpy as np
from typing import Dict, List, Optional
from model.discriminator.discriminator import TemporalDiscriminator
from model.CAT import ContextAwareTransformer
from model.TranSformer import S4SceneModel



def build_center_reg_labels_from_relative_targets(targets_list):
    """
    从你当前 dataset 的 targets（包含 'boundaries' in [0,1]）构造 K=1 的监督：
    - cls_label: (B,) 0/1，窗口内是否存在至少一个 GT 边界
    - center_gt: (B,) ∈ [0,1]，若存在多个 GT，选离 0.5 最近的一个作为该窗的监督目标
    - has_pos:   (B,) bool，正样本掩码
    - vid_ids:   list[str]，用于汇总时写 submission
    """
    B = len(targets_list)
    device = (targets_list[0]['boundaries'].device
              if torch.is_tensor(targets_list[0]['boundaries'])
              else torch.device('cpu'))

    cls_label = torch.zeros(B, dtype=torch.float32, device=device)
    center_gt = torch.full((B,), 0.5, dtype=torch.float32, device=device)  # 默认占位 0.5
    has_pos = torch.zeros(B, dtype=torch.bool, device=device)
    vid_ids = []

    for i, t in enumerate(targets_list):
        # video id（注意你这里是 'video_id': Tensor([vid_idx])）
        vid_tensor = t.get('video_id', None)
        vid = int(vid_tensor.item()) if torch.is_tensor(vid_tensor) else (vid_tensor if vid_tensor is not None else i)
        vid_ids.append(str(vid))

        y = t['boundaries']  # Tensor of shape (#gt,)
        if y.numel() > 0:
            cls_label[i] = 1.0
            # 选离 0.5 最近的 GT 作为该窗的监督目标
            idx = torch.argmin(torch.abs(y - 0.5))
            center_gt[i] = y[idx]
            has_pos[i] = True
        else:
            cls_label[i] = 0.0
            # center_gt 保持 0.5（不计入损失）

    return cls_label, center_gt, has_pos, vid_ids


@torch.no_grad()
def map_center_to_abs_frames(center_pred, base, window_size, interval, locations=None):
    """
    将窗口内相对坐标中心 y ∈ [0,1] 映射到绝对帧索引。
    简化一致间隔：abs_frame = base + y * (window_size * interval)
    若给定 locations，可换成插值到真实帧号（更精确）。
    输入:
      center_pred: (B,) in [0,1]
      base: (B,) 或 标量，窗口起始绝对帧号
    返回:
      abs_frames: (B,) float
    """
    if locations is None:
        win_span = window_size * interval  # 帧
        abs_frames = base.float() + center_pred.float() * float(win_span)
        return abs_frames
    else:
        # 若 locations: (B, T, 1) 表示真实帧号或索引，可用线性插值
        # y ∈ [0,1] → t = y * (T-1)，再用 locations[b, :, 0] 做插值
        B, T, _ = locations.shape
        t = center_pred.clamp(0, 1) * (T - 1)
        abs_frames = []
        for b in range(B):
            xs = torch.arange(T, device=locations.device, dtype=locations.dtype)
            ys = locations[b, :, 0]  # 假设这是绝对帧索引（或可换成映射）
            tb = t[b]
            # 线性插值
            lo = torch.clamp(tb.floor().long(), 0, T - 1)
            hi = torch.clamp(lo + 1, 0, T - 1)
            wb = tb - lo.float()
            abs_b = ys[lo] * (1 - wb) + ys[hi] * wb
            abs_frames.append(abs_b)
        return torch.stack(abs_frames, dim=0)


import torch
import torch.nn as nn
from tqdm import tqdm

def train_epoch(
    trainload,
    model,                 # 输出 (score_logit, center_pred)
    opti,
    lr_sh,
    gpu=0,
    pos_weight=None,       # 例如不平衡时用 torch.tensor([w]).to(gpu)
    lambda_reg=1.0,        # 回归损失权重

):
    model.train().cuda(gpu)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    l1 = nn.SmoothL1Loss(reduction='none')

    progress = tqdm(trainload, desc="Train(K=1, Center+Score)")

    for sample in progress:
        # 你的 collate_fn：locations, features, target_list, num_frames, base, coherence_scores
        locations, features, targets_list, num_frames, base, coherence = sample

        features = features.cuda(gpu, non_blocking=True)  # (B, L, N)

        # 从 targets['boundaries'] 构造监督
        cls_label, center_gt, has_pos, _ = build_center_reg_labels_from_relative_targets(targets_list)
        cls_label = cls_label.cuda(gpu, non_blocking=True)   # (B,)
        center_gt = center_gt.cuda(gpu, non_blocking=True)   # (B,)
        has_pos = has_pos.cuda(gpu, non_blocking=True)       # (B,)

        opti.zero_grad(set_to_none=True)

        score_logit, center_pred = model(features)           # (B,), (B,)
        loss_cls = F.binary_cross_entropy_with_logits(score_logit, cls_label.float())

        loss = loss_cls
        loss.backward()
        opti.step()
        if lr_sh is not None:
            lr_sh.step()

        progress.set_postfix(loss=float(loss.item()))
    return 1

from collections import defaultdict
import torch
from tqdm import tqdm


@torch.no_grad()
def test_epoch(testload, model, gpu=0):
    """推理事件数据集并收集边界预测。"""
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

        # 相对坐标插值得到绝对帧位置
        t = center_pred.clamp(0, 1) * (L - 1)
        lo = t.floor().long().clamp(0, L - 1)
        hi = (lo + 1).clamp(0, L - 1)
        w = (t - lo.float())
        rel_lo = loc[torch.arange(B), lo, 0]
        rel_hi = loc[torch.arange(B), hi, 0]
        rel_abs = rel_lo * (1 - w) + rel_hi * w
        abs_frames = base_b + rel_abs

        for i, tdict in enumerate(targets_list):
            vid_tensor = tdict.get("video_id", None)
            vid = (
                int(vid_tensor.item())
                if torch.is_tensor(vid_tensor)
                else (vid_tensor if vid_tensor is not None else i)
            )
            vid = str(vid)
            results_dict[vid].append(
                {
                    "boundaries": abs_frames[i : i + 1].detach().cpu(),
                    "scores": probs[i : i + 1].detach().cpu(),
                }
            )
    return dict(results_dict)



def main():
    # 创建参数对象
    args = argparse.Namespace()
    args.feature_path = "/data/shared_dataset/Kinetics/features"
    args.score_path = "/data/shared_dataset/Kinetics/data"
    args.annotation_path = "/data/shared_dataset/Kinetics/data"
    args.window_size = 21
    args.interval = 1
    args.device = 'cuda:0'
    args.output_dir = '/home/tianxiaoxuan/data/mamba/checkpoint_event_withoutLocalwithoutD'
    os.makedirs(args.output_dir, exist_ok=True)

    # 1) 构建数据
    train_dataset = build(split='train', args=args)
    val_dataset = build(split='val', args=args)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=256, shuffle=True, collate_fn=collate_fn, num_workers=0)
    n = len(val_dataset)
    val_subset = Subset(val_dataset, list(range(int(0.1 * n))))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 2) 构建无 Query 的 detector/head
    model = LocalUncertaintyAwareGraphAttentionLite(
    dim_in=2048,          # x 的通道维 Cx
    num_heads=8,         # 需满足 (dim_in + place_dim) % num_heads == 0
    window_size=5,       # 建议奇数：5/9/17
    sim_temperature=0.07,
    neighbor_temp=0.5,
    uncertainty_mode="variance",  # 或 "confidence"
    use_relative_pos_bias=True,
    norm="ln",           # "ln" 更稳，"bn" 需要更大 batch
    size=args.window_size
).to(args.device)
#     model = SimpleTransformerSceneModel().to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4,
    )
    lr_sh = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_decay_cosine(len(train_loader), len(train_loader) * 19)
    )

    # 4) 训练/验证循环
    epochs = 20
    for ep in range(epochs):
        train_epoch(
            trainload=train_loader,
            model=model, opti=optimizer, lr_sh=lr_sh,
            gpu=int(args.device.split(':')[-1]),
            lambda_reg=1.0
        )

        # # # # 保存
        # torch.save({'model': model.state_dict()}, os.path.join(args.output_dir, f'ckpt_ep{ep}.pth'))

        results_dict = test_epoch(
            val_loader, model,
            gpu=int(args.device.split(':')[-1]),
        )
        thresholds = [0.4,0.50]
        rel_d_list = (0.05,0.10)

        eval_res = evaluate_multi_thresholds(
            results_dict, anno_root_path=args.annotation_path,
            thresholds=thresholds,
            rel_dist_list=rel_d_list,
            step_frames=1,
            out_dir='multif-pred_outputs',
            write_pkls=False,  # 如需每个阈值生成 submission_thXX.pkl 可设 True
            debug=True,
        )

        global_avg = summarize_global_avg_over_thresholds(eval_res, d=0.10)
        print(f"[GLOBAL AVG over thresholds] F1@0.10={global_avg['f1']:.4f} "
              f"(Prec={global_avg['precision']:.4f}, Rec={global_avg['recall']:.4f})")

def evaluate_multi_thresholds(
        results, anno_root_path,
        thresholds=np.linspace(0.05, 0.50, 10),  # 0.05~0.50 共10个阈值
        rel_dist_list=(0.10),
        step_frames=1,
        out_dir='multif-pred_outputs',
        write_pkls=False,
        debug=True
):
    """
    一键：对多个 score 阈值，评测多个容忍半径 d，并返回每个阈值的 by-d 指标与平均。
    - results: 模型原始输出 map
    - thresholds: 想评测的一组分数阈值
    - rel_dist_list: 评测容忍半径列表（F1@0.05~0.50）
    - 返回: {th: {'by_thresh': {d: {...}}, 'avg': {...}}, ...}
    """
    seconds_map_by_thresh = save_submission_in_seconds_multi_thresh(
        results, anno_root_path, thresholds,
        step_frames=step_frames, out_dir=out_dir,
        write_pkls=write_pkls, debug=debug
    )
    eval_results = eval_by_frame_metric_from_seconds_map(
        anno_root_path, seconds_map_by_thresh,
        downsample=1, rel_dist_list=rel_dist_list,
        filter_low_consis=True, consis_th=0.3, debug=debug,
        output_file = 'null'
    )
    return eval_results
if __name__ == '__main__':
    main()

