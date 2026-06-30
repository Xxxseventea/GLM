import argparse
import os
from collections import defaultdict

import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data import Subset
from tqdm import tqdm

from tool.metric.kinetics_metric import save_submission_in_seconds_multi_thresh,eval_by_frame_metric_from_seconds_map,summarize_global_avg_over_thresholds
from dataset.kinetics.dataset import build, collate_fn  # 假设代码保存在某个文件
import os
import json
import pickle
import numpy as np

from model.lgss_event import LGSSEventDet 
from datetime import datetime

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
    # bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
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

        score_logit, _ = model(features, "event")           # (B,), (B,)
        loss_cls = F.binary_cross_entropy_with_logits(score_logit.squeeze(-1), cls_label.float())

        loss = loss_cls
        loss.backward()
        opti.step()
        

        progress.set_postfix(loss=float(loss.item()))
    if lr_sh is not None:
        lr_sh.step()
    return 1

from collections import defaultdict
import torch
from tqdm import tqdm

@torch.no_grad()
def test_epoch(testload, model, gpu=0):
    model.eval().cuda(gpu)
    results_dict = defaultdict(list)

    for batch in tqdm(testload, desc="Test collecting"):
        locations, features, targets_list, num_frames, base, coherence = batch
        B, L, _ = features.shape
        features = features.cuda(gpu, non_blocking=True)

        score_logit, center_pred = model(features, "event")
        probs = torch.sigmoid(score_logit)

        base_b = base.view(-1).to(features.device).float()
        loc = locations.to(features.device).float()

        t = center_pred.clamp(0, 1) * (L - 1)
        lo = t.floor().long().clamp(0, L - 1)
        hi = (lo + 1).clamp(0, L - 1)
        w = (t - lo.float())
        rel_lo = loc[torch.arange(B), lo, 0]
        rel_hi = loc[torch.arange(B), hi, 0]
        rel_abs = rel_lo * (1 - w) + rel_hi * w                  # (B,) 相对 base 的帧偏移
        abs_frames = base_b + rel_abs                            # (B,) 绝对帧（float）

        for i, tdict in enumerate(targets_list):
            vid_tensor = tdict.get('video_id', None)
            vid = int(vid_tensor.item()) if torch.is_tensor(vid_tensor) else (vid_tensor if vid_tensor is not None else i)
            vid = str(vid)
            results_dict[vid].append({
                'boundaries': abs_frames[i:i+1].detach().cpu(),  # Tensor([frame])
                'scores': probs[i:i+1].detach().cpu(),           # Tensor([score])
            })

    return dict(results_dict)

def to_serializable(obj):
    """
    递归把 numpy / torch / 标量等转成可写入 json 的类型
    """
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.float16)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    else:
        return obj

def save_json(data, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, ensure_ascii=False, indent=2)


def main():
    # 创建参数对象
    kinetics_dataset_path = "/data/Kinetics/"
    args = argparse.Namespace()
    args.feature_path = kinetics_dataset_path + "features"
    args.score_path = kinetics_dataset_path + "data"
    args.annotation_path = kinetics_dataset_path + "data"
    args.window_size = 21
    args.interval = 1
    args.device = 'cuda:0'
    args.output_dir = '<output_dir>'
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

    mlp_hid_dim=512
    mlp_out_dim=1
    # 2) 构建无 Query 的 detector/head
    
    model = LGSSEventDet(shot_num=20, place_feat_dim=2048, sim_channel=512)
    model = model.to(args.device)
#     model = SimpleTransformerSceneModel().to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-2, betas=(0.9, 0.98), weight_decay=5e-4,
    )
    lr_sh = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,milestones=[15]
    )

    # 4) 训练/验证循环
    epochs = 30
    bce_pos_weight = 5  # 如需正样本加权，设 torch.tensor([w]).to(args.device)
    for ep in range(epochs):
        train_epoch(
            trainload=train_loader,
            model=model, opti=optimizer, lr_sh=lr_sh,
            gpu=int(args.device.split(':')[-1]),
            pos_weight=bce_pos_weight, lambda_reg=1.0
        )

        # # # 保存
        torch.save({'model': model.state_dict()}, os.path.join(args.output_dir, f'ckpt_ep{ep}.pth'))

        # 验证：收集窗口候选，转秒并评测
        results_dict = test_epoch(
            val_loader, model,
            gpu=int(args.device.split(':')[-1]),
        )
        thresholds = [0.10,  0.20,  0.30, 0.40,  0.50]
        rel_d_list = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

        # eval_res = evaluate_multi_thresholds(
        #     results_dict, anno_root_path=args.annotation_path,
        #     thresholds=thresholds,
        #     rel_dist_list=rel_d_list,
        #     step_frames=1,
        #     out_dir='multif-pred_outputs',
        #     write_pkls=False,  # 如需每个阈值生成 submission_thXX.pkl 可设 True
        #     debug=True
        # )
        eval_res = evaluate_multi_thresholds(
                results_dict,
                anno_root_path=args.annotation_path,
                thresholds=[0.10, 0.20, 0.30, 0.40, 0.50],
                rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
                step_frames=1,
                log_tag='event_eval',
            )

        event_eval_record = {
                "stage": "event_training",
                "epoch": ep,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_videos": len(results_dict),
                "results": eval_res,
        }

        save_json(
                event_eval_record,
                os.path.join(args.output_dir, f"event_eval_ep{ep}.json")
            )


        global_avg = summarize_global_avg_over_thresholds(eval_res, d=0.10)
        print(f"[GLOBAL AVG over thresholds] F1@0.10={global_avg['f1']:.4f} "
              f"(Prec={global_avg['precision']:.4f}, Rec={global_avg['recall']:.4f})")


def evaluate_multi_thresholds(
    results,
    anno_root_path,
    thresholds=None,
    rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    step_frames=1,
    log_tag="event_eval",
):
    """多阈值评测事件检测性能"""
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]

    seconds_map_by_thresh = save_submission_in_seconds_multi_thresh(
        results,
        anno_root_path,
        thresholds,
        step_frames=step_frames,
        out_dir=f"{log_tag}_pred_outputs",
        write_pkls=False,
        debug=True,
    )

    eval_results = eval_by_frame_metric_from_seconds_map(
        anno_root_path,
        seconds_map_by_thresh,
        downsample=1,
        rel_dist_list=rel_dist_list,
        filter_low_consis=True,
        consis_th=0.3,
        debug=True,
    )

    global_avg = summarize_global_avg_over_thresholds(eval_results, d=0.10)
    print(
        f"[GLOBAL AVG] F1@0.10={global_avg['f1']:.4f} "
        f"(Prec={global_avg['precision']:.4f}, Rec={global_avg['recall']:.4f})"
    )
    return eval_results
if __name__ == '__main__':
    main()

