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
from typing import Dict, Any

from model.resnet50_1d import resnet50_1d 
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
    epoch,
    gpu=0,
    pos_weight=None,       # 例如不平衡时用 torch.tensor([w]).to(gpu)
    lambda_reg=1.0,        # 回归损失权重

):
    model.train().cuda(gpu)
    # bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    l1 = nn.SmoothL1Loss(reduction='none')

    progress = tqdm(trainload, desc="Train(K=1, Center+Score)")
    num_updates = epoch * len(trainload)
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

        score_logit, _ = model(features)           # (B,), (B,)
        loss_cls = F.binary_cross_entropy_with_logits(score_logit.squeeze(-1), cls_label.float())

        loss = loss_cls
        loss.backward()
        opti.step()
        num_updates += 1
        lr_sh.step_update(num_updates=num_updates, metric=loss.item())
        

        progress.set_postfix(loss=float(loss.item()))
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

        score_logit, center_pred = model(features)
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

class Scheduler:
    """ Parameter Scheduler Base Class
    A scheduler base class that can be used to schedule any optimizer parameter groups.
    Unlike the builtin PyTorch schedulers, this is intended to be consistently called
    * At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
    * At the END of each optimizer update, after incrementing the update count, to calculate next update's value
    The schedulers built on this should try to remain as stateless as possible (for simplicity).
    This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
    and -1 values for special behaviour. All epoch and update counts must be tracked in the training
    code and explicitly passed in to the schedulers on the corresponding step or step_update call.
    Based on ideas from:
     * https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler
     * https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize: bool = True) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.metric = None  # any point to having this for all?
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_epoch_values(self, epoch: int):
        return None

    def get_update_values(self, num_updates: int):
        return None

    def step(self, epoch: int, metric: float = None) -> None:
        self.metric = metric
        values = self.get_epoch_values(epoch)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None):
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
            if apply_noise:
                g = torch.Generator()
                g.manual_seed(self.noise_seed + t)
                if self.noise_type == 'normal':
                    while True:
                        # resample if noise out of percent limit, brute force but shouldn't spin much
                        noise = torch.randn(1, generator=g).item()
                        if abs(noise) < self.noise_pct:
                            break
                else:
                    noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
                lrs = [v + v * noise for v in lrs]
        return lrs
    

class StepLRScheduler(Scheduler):
    """
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 decay_t: float,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.decay_t = decay_t
        self.decay_rate = decay_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [v * (self.decay_rate ** (t // self.decay_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

def main():
    # 创建参数对象
    kinetics_dataset_path = "data/Kinetics/"
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
    
    # 实例化 UAG：两层、窗口半径 6（窗口长度 13）、每步保留 8 个邻居
    model = resnet50_1d(in_channels=2048, n_classes=1, short_seq=True)

    model = model.to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=0.01)
    lr_sh = StepLRScheduler(optimizer=optimizer, warmup_lr_init=0.0001,decay_t=10, decay_rate=0.1, warmup_t=0)



    # 4) 训练/验证循环
    epochs = 30
    bce_pos_weight = 5  # 如需正样本加权，设 torch.tensor([w]).to(args.device)
    for ep in range(epochs):
        train_epoch(
            trainload=train_loader,
            model=model, opti=optimizer, lr_sh=lr_sh,
            gpu=int(args.device.split(':')[-1]),
            pos_weight=bce_pos_weight, lambda_reg=1.0,
            epoch=ep
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

        max_th = max(eval_res.keys())
        f1 = eval_res[max_th]['avg']['f1']
        lr_sh.step(ep+1, f1)
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

