
"""
两任务持续学习的简化训练脚本（基于 NSCL / BECAME）。
流程：先训练 Task1 → 计算 SVD 子空间 / EWC / Fisher → 训练 Task2（稳定容器 + 蒸馏 + Fisher 融合）
依赖本仓库的 models 与 optim（自定义带 SVD 投影的 Adam）。

用法：
    替换下方 `build_task_loaders()` 内部的数据集为你自己的两个数据集；
    运行：python train_two_tasks.py
"""

import os
import re
import copy
import random
from types import MethodType
from collections import defaultdict
import argparse
from datetime import datetime

import numpy as np
import torch
import json as js
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

from model.backbone.RelationNet_became import LocalUncertaintyAwareGraphAttentionLite
from model.detector.scene_detector import MlpHead
from model.detector.event_detector import FrameWiseHead
from dataset.movienet.load_movienet_server import MovieNetDataset, read_pkl, read_pkl2
from dataset.kinetics.dataset import KineticsGEBD, collate_fn
from tool.metric.scene_metric import metric
import optim as custom_optim  # 自定义 Adam（支持 SVD 子空间投影）
from tool.metric.kinetics_metric import (
    save_submission_in_seconds_multi_thresh,
    eval_by_frame_metric_from_seconds_map,
    summarize_global_avg_over_thresholds,
)

# =============================================================
# 1. 配置
# =============================================================
CFG = {
    'device':            'cuda:0' if torch.cuda.is_available() else 'cpu',
    'seed':              0,
    'batch_size':        128,
    'num_workers':       0,
    "T":                 21, 

    'task1_num_classes': 1,
    'task2_num_classes': 1,

    # epoch 设置
    'task1_epochs':        5,   # Task1 总训练轮数
    'task2_stage1_epochs': 2, # Task2 稳定容器阶段（SVD 投影）
    'task2_stage2_epochs': 2, # Task2 可塑性 + 蒸馏 + 融合阶段

    'model_lr':          5e-4,
    'svd_lr':            5e-4,
    'head_lr':           5e-4,
    'bn_lr':             5e-4,
    'model_weight_decay': 1e-5,
    'svd_thres':         1.0,

    'reg_coef':          100.0,

    'use_distill':       True,
    'distill_coef':      1.0,

    'kinetics_dataset_path': 'data/Kinetics/',
    'kinetics_data': 'data',
    'movienet_dataset_path': 'data/MovieNet/',
    'split_path': 'split318.json',
    'modalA_path': 'ImageNet_shot.pkl',
    'modalB_path': 'Places_shot.pkl', 
    'seg_sz': 20,
    'label_path': 'label_endShot.pkl',
    'task_order': ['scene', 'event'],   # 先scene再event
    'save_path': '<save_dir>'
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--order', nargs=2, choices=['scene', 'event'],
                   default=CFG['task_order'],
                   help="例：--order event scene 或 --order scene event")
    return p.parse_args()

class WithTaskName(Dataset):
    """将 (img, target) → (img, target, task_name)；模型多头路由需要任务名。"""
    def __init__(self, dataset, task_name):
        self.dataset = dataset
        self.task_name = task_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return (*sample, self.task_name)


def build_task_loaders():
    args = argparse.Namespace()
    feature_path = CFG["kinetics_dataset_path"] + "features"
    annotation_path = CFG["kinetics_dataset_path"] + "data"
    score_path = CFG["kinetics_dataset_path"] + "data"
    args.feature_path = feature_path
    args.annotation_path = annotation_path
    args.score_path = score_path
    args.window_size = CFG["T"]
    args.interval = 1
    args.device = "cuda:0"

    with open(CFG["movienet_dataset_path"] + CFG["split_path"], 'r') as f:
        data = js.load(f)
        splitSet = data['train']  + data['val']

    modalA_feat = read_pkl2(CFG["movienet_dataset_path"] + CFG["modalA_path"])
    modalB_feat = read_pkl2(CFG["movienet_dataset_path"] + CFG["modalB_path"])
    seg_sz = CFG["seg_sz"]
    labels = read_pkl(CFG["movienet_dataset_path"] + CFG["label_path"])
    train_scene_dataset = MovieNetDataset(labels, modalA_feat, modalB_feat, splitSet, seg_sz, "train", None)
    # train_scene_dataset = WithTaskName(train_scene_dataset, 'scene')
    train_scene_loader = torch.utils.data.DataLoader(train_scene_dataset, batch_size=CFG["batch_size"],
                                                 shuffle=True, drop_last=True ,num_workers=0)

    
    feature_folder = Path(feature_path)
    score_path = Path(score_path)
    anno_file = Path(annotation_path)

    train_event_dataset = KineticsGEBD(feature_folder, score_path, anno_file, "train", args)
    # train_event_dataset = WithTaskName(train_event_dataset, 'event')
    train_event_loader = torch.utils.data.DataLoader(
        train_event_dataset, batch_size=CFG['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=2,
        pin_memory=False, drop_last=True
    )


    test_scene_dataset = MovieNetDataset(labels, modalA_feat, modalB_feat, data["test"], seg_sz, "test", None)
    # test_scene_dataset = WithTaskName(test_scene_dataset,  'scene')
    test_scene_loader = torch.utils.data.DataLoader(test_scene_dataset, batch_size=CFG["batch_size"],
                                                 shuffle=False, drop_last=True, num_workers=0)


    test_dataset_event = KineticsGEBD(feature_folder, score_path, anno_file, "val", args)
    # test_dataset_event = WithTaskName(test_dataset_event,  'event')
    test_event_loader = torch.utils.data.DataLoader(
        test_dataset_event, batch_size=CFG['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=2,
        pin_memory=False, drop_last=False
    )

    dataloaders = {
        "event_train": train_event_loader,
        "event_test": test_event_loader,
        "scene_train": train_scene_loader,
        "scene_test": test_scene_loader,
    }
    # return train_event_loader, test_event_loader, train_scene_loader, test_scene_loader
    return dataloaders


# =============================================================
# 3. 构建多头模型（保留 features_last 便于蒸馏）
# =============================================================
def build_model(dim_in=2048, num_heads=8, window_size=15, size=21, mlp_hid_dim=512, mlp_out_dim=1, device="cuda:0"):
    # model = resnet18()
    model = LocalUncertaintyAwareGraphAttentionLite(
        dim_in=dim_in,
        num_heads=num_heads,
        window_size=window_size,
        sim_temperature=0.07,
        neighbor_temp=0.5,
        uncertainty_mode="variance",
        use_relative_pos_bias=True,
        norm="ln",
        size=size,
    )

    model.detector = nn.ModuleDict({
       "scene": MlpHead(in_dim=2176, hid_dim=mlp_hid_dim, out_dim=mlp_out_dim),
        "event": FrameWiseHead(in_features=2176),
    })
    def new_logits(self, x):
        outputs = {t: f(x) for t, f in self.detector.items()}
        outputs['ALL'] = torch.cat([outputs[t] for t in self.detector.keys()], dim=1)
        return outputs
    model.logits = MethodType(new_logits, model)
    return model.to(device)


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
        js.dump(to_serializable(data), f, ensure_ascii=False, indent=2)

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

SVD_EXCLUDE = {'uncert_proj.weight', 'mamba.final_fusion.weight'}

# =============================================================
# 4. 训练器
# =============================================================
class TwoTaskTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg['device']
        self.model = build_model(device=self.device)
        self.pre_model = None
        self.m_model   = None

        self.reg_params = {n: p for n, p in self.model.named_parameters() if 'bn' in n}
        self.reg_terms  = {}

        self.fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                       if p.requires_grad and not re.match(r'^detector', n)}

        self.fea_in = defaultdict(dict)
        self._hook_handles = []

        self.criterions = {
            'scene': nn.BCEWithLogitsLoss(),
            'event': nn.BCEWithLogitsLoss(),
        }
        self.task_count = 0
        self.is_distill = False

        self._build_optimizer(task_idx=0)
        self.best_metric = None

    def save_checkpoint(self, epoch, task_name, stage='', is_best=False, extra=None):
        """保存完整的 checkpoint，支持恢复训练"""
        ckpt_dir = Path(self.cfg['save_path']) / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cfg': self.cfg,
            'task_count': self.task_count,
            'best_metric': self.best_metric,
            'reg_terms': self.reg_terms,          # EWC 重要性矩阵
            'fisher': {k: v.cpu() for k, v in self.fisher.items()},  # Fisher 信息矩阵
        }
        if extra:
            checkpoint.update(extra)

        if stage:
            filename = f'{task_name}_{stage}_ep{epoch}.pth'
        else:
            filename = f'{task_name}_ep{epoch}.pth'

        filepath = ckpt_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")

        if is_best:
            best_path = ckpt_dir / f'{task_name}_best.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved: {best_path}")

    def _build_optimizer(self, task_idx):
        task_name = self.cfg['task_order'][task_idx]
        lr_feat = self.cfg['model_lr'] if task_idx == 0 else self.cfg['svd_lr']

        fea_params, fea_ids = [], set()
        for n, m in self.model.named_modules():
            if re.match(r'^detector', n):
                continue
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                if f'{n}.weight' in SVD_EXCLUDE:   # ← 跳过，让它落到 other_params
                    continue
                if m.weight.requires_grad:
                    fea_params.append(m.weight)
                    fea_ids.add(id(m.weight))   # ← 注意：不收 bias

        cls_params = list(self.model.detector[task_name].parameters())
        cls_ids = {id(p) for p in cls_params}

        other_params = [p for _, p in self.model.named_parameters()
                    if p.requires_grad
                    and id(p) not in fea_ids
                    and id(p) not in cls_ids]

        arg = {
            'params': [
                {'params': fea_params,   'svd': True, 'lr': lr_feat,
                'thres': self.cfg['svd_thres']},
                {'params': cls_params,   'weight_decay': 0.0, 'lr': self.cfg['head_lr']},
                {'params': other_params, 'lr': self.cfg['bn_lr']},
            ],
        'lr': self.cfg['model_lr'],
        'weight_decay': self.cfg['model_weight_decay'],
        }
        self.optimizer = custom_optim.Adam(**arg)

    def _reg_loss(self):
        loss = 0.0
        for n, p in self.reg_params.items():
            imp   = torch.cat(self.reg_terms['importance'][n], dim=0)
            old_p = torch.cat(self.reg_terms['task_param'][n], dim=0)
            new_p = p.unsqueeze(0).expand(old_p.shape)
            loss = loss + (imp * (new_p - old_p) ** 2).sum()
        return loss

    def _full_loss(self, score_logit, target, task_name, regularize=True):
        loss = self.criterions[task_name](score_logit, target)
        if regularize and len(self.reg_terms) > 0:
            loss = loss + self.cfg['reg_coef'] * self._reg_loss()
        return loss

    def _train_epoch_event(self, loader, task_name):
        self.model.train()
        tot, n = 0.0, 0
        for _, features, targets_list, _, _, _  in loader:
            cls_label, center_gt, has_pos, _ = build_center_reg_labels_from_relative_targets(targets_list)
            cls_label = cls_label.to(self.device)   # (B,)
            center_gt = center_gt.to(self.device)   # (B,)
            has_pos = has_pos.to(self.device)       # (B,)

            features = features.to(self.device)

            score_logit, center_pred = self.model(features, task_name)
            loss = self._full_loss(score_logit, cls_label.float(), task_name=task_name)

            if self.task_count > 0 and not self.optimizer.switch and self.is_distill:
                fa = self.model.features_last(features)
                with torch.no_grad():
                    fb = self.pre_model.features_last(features)
                loss = loss + self.cfg['distill_coef'] * (fa - fb).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tot += loss.item() * len(features); n += len(cls_label)
        return tot / max(n, 1)

    def _train_epoch_scene(self, loader, task_name):
        self.model.train()
        tot, n = 0.0, 0
        for _, img_s, plc_s, label_s, pos, _, _ in loader:
            img_s = img_s.to(self.device)
            plc_s = plc_s.to(self.device)
            label_s = label_s.to(self.device).float()

            w = 0.7
            x_new = w * img_s + (1 - w) * plc_s
            logits_new, _ = self.model(x_new, task_name)
            logits_new = logits_new.squeeze(-1)
            loss = self._full_loss(logits_new, label_s.float(), task_name=task_name)

            # 蒸馏：仅 Task2 的 switch=False 阶段开启
            if self.task_count > 0 and not self.optimizer.switch and self.is_distill:
                fa = self.model.features_last(x_new)
                with torch.no_grad():
                    fb = self.pre_model.features_last(x_new)
                loss = loss + self.cfg['distill_coef'] * (fa - fb).pow(2).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            tot += loss.item() * len(x_new); n += len(label_s)
        return tot / max(n, 1)


    @torch.no_grad()
    def validate(self, loader, task_name, ep=0):
        self.model.eval()
        if task_name == "event":
            results = self.test_epoch_event(testload=loader, model=self.model, task_name=task_name)
            print(f"✅ Collected predictions for {len(results)} videos")

            # 多阈值评测
            eval_res = evaluate_multi_thresholds(
                results,
                anno_root_path=str(CFG['kinetics_dataset_path']+CFG['kinetics_data']),
                thresholds=[0.10, 0.20, 0.30, 0.40, 0.50],
                rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
                step_frames=1,
                log_tag='event_eval',
            )
            event_eval_record = {
                "stage": "phase2_event_training",
                "epoch": ep,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_videos": len(results),
                "results": eval_res,
            }

            save_json(
                event_eval_record,
                os.path.join(CFG["save_path"], f"event_eval_ep{ep}.json")
            )

            torch.save({"model": self.model.state_dict(), "epoch": ep},
                   os.path.join(CFG["save_path"], f'event_ep{ep}.pth'))

            print("\n" + "="*60)

        elif task_name == "scene":
            best_path = os.path.join(CFG["save_path"], "scene_best.pth")
            met, moviePL = self.eval_epoch_scene(test_loader=loader, model=self.model, task_name=task_name)
            print(f"[Epoch {ep}] metric = {met}")
            scene_eval_record = {
                "stage": "phase1_scene_training",
                "epoch": ep,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metric": met,
                "moviePL": moviePL,
            }

            save_json(
                scene_eval_record,
                os.path.join(CFG["save_path"], f"scene_eval_ep{ep}.json")
            )

            torch.save({"model": self.model.state_dict(), "epoch": ep},
               os.path.join(CFG["save_path"], f'scene_ep{ep}.pth'))

            if isinstance(met, dict):
                key = next(iter(met.keys()))
                score = float(met[key])
                if (self.best_metric is None) or (score > self.best_metric):
                    self.best_metric = score
                    torch.save({"model": self.model.state_dict(), "epoch": ep}, best_path)
                    print(f"New best ({key}={score:.4f}) saved -> {best_path}")

    @torch.no_grad()
    def eval_epoch_scene(self, test_loader, model, task_name, device='cuda:0'):
        model.eval()
        predlist, labelist, pathlist, idlist = [], [], [], []
        for sample in tqdm(test_loader, desc="Eval(Scene)"):
            names, img_ctx, plc_ctx, label, pos, ids, ind, = sample
            img_ctx = img_ctx.to(device, non_blocking=True)
            plc_ctx = plc_ctx.to(device, non_blocking=True)
            w = 0.7
            x = w * img_ctx + (1 - w) * plc_ctx

            logits, center = model(x, task_name)
            probs = center.squeeze().cpu().numpy()

            predlist.append(probs)
            labelist.append(label.numpy())
            pathlist.append(names)
            idlist.append(np.asarray(ind))
        met, moviePL = metric(pathlist, predlist, labelist)
        return met, moviePL


    @torch.no_grad()
    def test_epoch_event(self, testload, model, task_name, gpu=0):
        model.eval().cuda(gpu)
        results_dict = defaultdict(list)

        for batch in tqdm(testload, desc="Evaluate(Event Dataset)"):
            locations, features, targets_list, num_frames, base, coherence = batch

            B, L, _ = features.shape
            features = features.cuda(gpu, non_blocking=True)

            score_logit, center_pred = model(features, task_name)
            probs = torch.sigmoid(score_logit)

            base_b = base.view(-1).to(features.device).float()
            loc = locations.to(features.device).float()

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
                results_dict[vid].append({
                    "boundaries": abs_frames[i : i + 1].detach().cpu(),
                    "scores": probs[i : i + 1].detach().cpu(),
                })
        return dict(results_dict)
    def _cov_hook(self, module, fea_in, fea_out):
        if isinstance(module, nn.Linear):
            x = fea_in[0]                          # (B, *, C_in)，可能是 2D 或 3D+
            if x.dim() > 2:
                x = x.mean(dim=0)                  # 先对 batch 取均值省内存 → (*, C_in)
            else:
                x = x.mean(dim=0, keepdim=True)    # 2D 场景保持原行为 → (1, C_in)
            x = x.reshape(-1, x.shape[-1])         # 统一展平为 (N, C_in)
            self._update_cov(x, module.weight)
        elif isinstance(module, nn.Conv2d):
            x = F.unfold(torch.mean(fea_in[0], 0, True),
                     kernel_size=module.kernel_size,
                     padding=module.padding, stride=module.stride)
            x = x.permute(0, 2, 1).reshape(-1, x.shape[1])
            self._update_cov(x, module.weight)

    def _update_cov(self, x, key):
        cov = x.t() @ x
        if len(self.fea_in[key]) == 0:
            self.fea_in[key] = cov
        else:
            self.fea_in[key] = self.fea_in[key] + cov

    @torch.no_grad()
    def build_svd_transforms(self, loader, task_name):
        self.fea_in = defaultdict(dict)
        modules = [m for n, m in self.model.named_modules()
               if isinstance(m, (nn.Linear, nn.Conv2d))
               and not re.match(r'^detector', n)
               and f'{n}.weight' not in SVD_EXCLUDE]
        self._hook_handles = [m.register_forward_hook(self._cov_hook) for m in modules]
        for names, img_ctx, plc_ctx, label, pos, ids, ind in loader:
            img_ctx = img_ctx.to(self.device, non_blocking=True)
            plc_ctx = plc_ctx.to(self.device, non_blocking=True)
            w = 0.7
            x = w * img_ctx + (1 - w) * plc_ctx
            self.model.forward(x.to(self.device), task_name=task_name)
        svd_id2name = {}
        for g in self.optimizer.param_groups:
            if g.get('svd', False):
                for p in g['params']:
                    svd_id2name[id(p)] = '<unknown>'
        for n, p in self.model.named_parameters():
            if id(p) in svd_id2name:
                svd_id2name[id(p)] = n

        recorded_ids = {id(k) for k, v in self.fea_in.items() if torch.is_tensor(v)}
        missing_names = [name for pid, name in svd_id2name.items() if pid not in recorded_ids]
        assert not missing_names, '两边过滤器没对齐，见上面的名字'

        self.optimizer.get_eigens(self.fea_in)
        self.optimizer.get_transforms()
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []
        torch.cuda.empty_cache()

    def store_ewc(self, loader, task_name):
        if not self.reg_params:
            return
        imp = {n: torch.zeros_like(p) for n, p in self.reg_params.items()}
        self.model.eval()
        for _, features, targets_list, _, _, _ in loader:
            features = features.to(self.device)
            score_logit, _ = self.model(features, task_name)
            cls_label, center_gt, has_pos, _ = build_center_reg_labels_from_relative_targets(targets_list)
            cls_label = cls_label.to(self.device)   # (B,)
            center_gt = center_gt.to(self.device)   # (B,)
            has_pos = has_pos.to(self.device)       # (B,)

            pseudo = (torch.sigmoid(score_logit) > 0.5).float().detach()

            loss = self._full_loss(score_logit, pseudo, regularize=False, task_name=task_name)
            self.model.zero_grad()
            loss.backward()
            for n in imp:
                if self.reg_params[n].grad is not None:
                    imp[n] += (self.reg_params[n].grad ** 2) * len(features) / len(loader)

        if not self.reg_terms:
            self.reg_terms = {'importance': defaultdict(list),
                              'task_param': defaultdict(list)}
        for n, p in self.reg_params.items():
            self.reg_terms['importance'][n].append(imp[n].unsqueeze(0))
            self.reg_terms['task_param'][n].append(p.detach().clone().unsqueeze(0))

    def compute_fisher(self, loader, task_name):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()
                  if p.requires_grad and not re.match(r'^detector', n)}
        self.model.train()
        if task_name == "event":
            for _, features, targets_list, _, _, _ in loader:
                features = features.to(self.device)
                score_logit, _ = self.model(features, task_name)
                pseudo = (torch.sigmoid(score_logit) > 0.5).float().detach()
                loss = self._full_loss(score_logit, pseudo, regularize=False, task_name=task_name)
                self.optimizer.zero_grad()
                loss.backward()
                for n, p in self.model.named_parameters():
                    if p.grad is not None and n in fisher:
                        fisher[n] += p.grad.pow(2) * len(features)
        elif task_name == "scene":
            for _, img_s, plc_s, label_s, pos, _, _ in loader:
                img_s = img_s.to(self.device)
                plc_s = plc_s.to(self.device)
                label_s = label_s.to(self.device).float()

                w = 0.7
                x_new = w * img_s + (1 - w) * plc_s
                logits_new, _ = self.model(x_new, task_name)
                logits_new = logits_new.squeeze(-1)
                pseudo = (torch.sigmoid(logits_new) > 0.5).float().detach()
                loss = self._full_loss(logits_new, pseudo, regularize=False, task_name=task_name)

                self.optimizer.zero_grad()
                loss.backward()
                for n, p in self.model.named_parameters():
                    if p.grad is not None and n in fisher:
                        fisher[n] += p.grad.pow(2) * len(img_s)
        n_samples = len(loader.dataset)
        return {n: p / n_samples for n, p in fisher.items()}

    def update_fisher(self, loader, task_name):
        cur = self.compute_fisher(loader, task_name)
        for n in self.fisher:
            self.fisher[n] += cur[n]
        return cur

    def fisher_merge(self, old_state, cur_state, loader, task_name):
        cur_fisher = self.compute_fisher(loader, task_name)
        mole, demo = 0.0, 0.0
        for k in self.fisher.keys():
            diff2 = (cur_state[k] - old_state[k]).pow(2)
            mole += (cur_fisher[k] * diff2).sum().item()
            demo += ((self.fisher[k] + cur_fisher[k]) * diff2).sum().item()
        coef = mole / max(demo, 1e-12)
        print(f'[Fisher merge] coef={coef:.4f}')
        merged = {}
        for k in old_state.keys():
            if k in cur_state:
                merged[k] = old_state[k] * (1 - coef) + cur_state[k] * coef
            else:
                merged[k] = old_state[k]
        return merged

    def train_task1(self, train_loader, val_loader):
        print('\n===== Task 1 =====')
        task_name = self.cfg['task_order'][0]
        self.optimizer.switch = True
        for ep in tqdm(range(self.cfg['task1_epochs']), desc='Task1'):
            # FIXME: 调整为scene的训练
            # loss = self._train_epoch_event(train_loader, task_name)
            loss = self._train_epoch_scene(train_loader, task_name=task_name)
            # FIXME: 检查下第一个顺序训scene是否需要有多阈值
            self.validate(val_loader, task_name)
            self.save_checkpoint(ep, task_name, stage='phase1', is_best=False)

        self.task_count = 1
        self._build_optimizer(task_idx=1)
        with torch.no_grad():
            self.build_svd_transforms(train_loader, task_name)
        self.store_ewc(train_loader, task_name)
        self.update_fisher(train_loader, task_name)

    def train_task2(self, train_loader, val_loader, val_loader_task1=None):
        print('\n===== Task 2 =====')
        name1, name2 = self.cfg['task_order']
        # 冻结 pre_model（蒸馏与融合都要用）
        self.pre_model = copy.deepcopy(self.model).to(self.device).eval()
        for p in self.pre_model.parameters():
            p.requires_grad = False

        self.optimizer.switch = True
        self.is_distill = False
        for ep in tqdm(range(self.cfg['task2_stage1_epochs']), desc='Task2-S1'):
            # FIXME: 调整为event的训练
            # loss = self._train_epoch_scene(train_loader, task_name=name2)
            loss = self._train_epoch_event(train_loader, name2)
            self.save_checkpoint(ep, name2, stage='phase1', is_best=False)
        self.m_model = copy.deepcopy(self.model.state_dict())
        self.save_checkpoint(self.cfg['task2_stage1_epochs'], name2, stage='stage1_end')

        if self.cfg['task2_stage2_epochs'] > 0:
            self.model.load_state_dict(self.pre_model.state_dict())
            self._build_optimizer(task_idx=1)
            self.optimizer.switch = False
            self.is_distill = self.cfg['use_distill']
            for ep in tqdm(range(self.cfg['task2_stage2_epochs']), desc='Task2-S2'):
                # FIXME: 调整为event的训练
                self._train_epoch_event(train_loader, name2)
                self.save_checkpoint(ep, name2, stage='stage2')
                # self._train_epoch_scene(train_loader, task_name=name2)
            cur_state = copy.deepcopy(self.model.state_dict())
            merged = self.fisher_merge(self.m_model, cur_state, train_loader, name2)
            self.model.load_state_dict(merged)
            self.save_checkpoint(self.cfg['task2_stage2_epochs'], name2, stage='final_merged', is_best=True)
        else:
            self.model.load_state_dict(self.m_model)
            self.save_checkpoint(0, name2, stage='only_stage1', is_best=True)

        self.update_fisher(train_loader, name2)

        # FIXME: 检查下第一个顺序训完scene后验证event是否需要有多阈值
        self.validate(val_loader, name2)
        if val_loader_task1 is not None:
            # FIXME: 检查下验证scene是否需要有多阈值
            self.validate(val_loader_task1, name1)
            
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    CFG['task_order'] = args.order

    set_seed(CFG['seed'])
    os.makedirs(CFG['save_path'], exist_ok=True)

    loaders = build_task_loaders()
    name1, name2 = CFG['task_order']
    tr1, va1 = loaders[f'{name1}_train'], loaders[f'{name1}_test']
    tr2, va2 = loaders[f'{name2}_train'], loaders[f'{name2}_test']

    trainer = TwoTaskTrainer(CFG)
    trainer.train_task1(tr1, va1)
    trainer.train_task2(tr2, va2, val_loader_task1=va1)

    torch.save(trainer.model.state_dict(),
               os.path.join(CFG['save_path'], 'final.pth'))
    print(f'Saved to {CFG["save_path"]}/final.pth')


if __name__ == '__main__':
    main()
