# main_continual.py
"""
持续学习主脚本：
  Task1: scene boundary detection (MovieNet)
  Task2: event boundary detection (KineticsGEBD)

特性：
  1) WSN 子网络持续学习（每任务独立 mask，旧任务参数冻结）
  2) 每个 epoch 保存 checkpoint
  3) 每个任务训完后保存 best.pth，下个任务从 best 继续学
  4) 所有任务训完后做最终联合评测，结果存 json
  5) 评估结果 / acc matrix 全部存 json
"""
import os
import json as js
import argparse
from copy import deepcopy
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from model.backbone.subnet import SubnetLinear
from model.backbone.RelationNet_PGM import LocalUncertaintyAwareGraphAttentionLite


# ⚠️ 替换成你的实际 import
from dataset.movienet.load_movienet_server import MovieNetDataset, read_pkl, read_pkl2
from dataset.kinetics.dataset import KineticsGEBD, collate_fn
from tool.metric.scene_metric import metric
from tool.metric.kinetics_metric import (
    save_submission_in_seconds_multi_thresh,
    eval_by_frame_metric_from_seconds_map,
    summarize_global_avg_over_thresholds,
)

# ============================================================
# 配置 (你可以改成从 yaml 读)
# ============================================================

TASKS = ['scene', 'event']  # 任务顺序，可改
CFG = {
    'device':            'cuda:0' if torch.cuda.is_available() else 'cpu',
    'seed':              0,
    'batch_size':        128,
    'num_workers':       0,
    "T":                 21, 

    # 两任务的类别数（用于建立两个分类头）
    'task1_num_classes': 10,
    'task2_num_classes': 10,

    # epoch 设置
    'task1_epochs':        2,   # Task1 总训练轮数
    'task2_stage1_epochs': 5, # Task2 稳定容器阶段（SVD 投影）
    'task2_stage2_epochs': 2, # Task2 可塑性 + 蒸馏 + 融合阶段

    # 学习率
    'model_lr':          5e-4,
    'svd_lr':            5e-4,
    'head_lr':           5e-4,
    'bn_lr':             5e-4,
    'model_weight_decay': 1e-5,
    'svd_thres':         1.0,   # SVD 特征值保留阈值

    # 正则化
    'reg_coef':          100.0, # EWC on BN 的强度

    # 蒸馏
    'use_distill':       True,
    'distill_coef':      1.0,

    'kinetics_dataset_path': '/root/autodl-tmp/Kinetics/',
    'kinetics_data': 'data',
    'movienet_dataset_path': '/mnt/MovieNet/',
    'split_path': 'split318.json',
    'modalA_path': 'ImageNet_shot.pkl',
    'modalB_path': 'Places_shot.pkl', 
    'seg_sz': 20,
    'label_path': 'label_endShot.pkl',
    'task_order': ['scene', 'event'],   # 或 ['scene', 'event']
    'save_path': '/root/autodl-fs/PGM/scene2event'
}



# ============================================================
# JSON 工具
# ============================================================
def to_serializable(obj):
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
# ============================================================
# 数据加载
# ============================================================
from torch.utils.data import Subset

def build_loaders(args):
    """返回每个任务的 (train_loader, test_loader)"""
    # ---- scene (MovieNet) ----
    labels = read_pkl(args.labels)
    modalA_feat = read_pkl(args.modelA_feat)
    modalB_feat = read_pkl(args.modelB_feat)
    seg_sz = args.seg_sz
    with open(args.split_path, 'r') as f:
        data = js.load(f)
        splitSet = data['train']  + data['val']

    train_scene_dataset = MovieNetDataset(
        labels, modalA_feat, modalB_feat, splitSet, seg_sz, "train", None
    )
    test_scene_dataset = MovieNetDataset(
        labels, modalA_feat, modalB_feat, data["test"], seg_sz, "test", None
    )

    # ---- event ----
    feature_folder = Path(args.feature_path)
    score_path = Path(args.score_path)
    anno_file = Path(args.annotation_path)

    train_event_dataset = KineticsGEBD(feature_folder, score_path, anno_file, "train", args)
    test_event_dataset = KineticsGEBD(feature_folder, score_path, anno_file, "val", args)

    # ⭐ 调试模式：截断数据集
    if getattr(args, 'debug', False):
        print(f"\n🐛 DEBUG MODE: train={args.debug_train_samples}, test={args.debug_test_samples}")
        train_scene_dataset = Subset(train_scene_dataset,
                                      range(min(args.debug_train_samples, len(train_scene_dataset))))
        test_scene_dataset = Subset(test_scene_dataset,
                                     range(min(args.debug_test_samples, len(test_scene_dataset))))
        train_event_dataset = Subset(train_event_dataset,
                                      range(min(args.debug_train_samples, len(train_event_dataset))))
        test_event_dataset = Subset(test_event_dataset,
                                     range(min(args.debug_test_samples, len(test_event_dataset))))

    # ---- DataLoaders ----
    train_scene_loader = torch.utils.data.DataLoader(
        train_scene_dataset, batch_size=CFG["batch_size"],
        shuffle=True, drop_last=False, num_workers=0   # ⭐ debug 时 drop_last=False
    )
    test_scene_loader = torch.utils.data.DataLoader(
        test_scene_dataset, batch_size=CFG["batch_size"],
        shuffle=False, drop_last=False, num_workers=0
    )
    train_event_loader = torch.utils.data.DataLoader(
        train_event_dataset, batch_size=CFG['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=0 if args.debug else 2,  # ⭐ debug 时 0 worker
        drop_last=False
    )
    test_event_loader = torch.utils.data.DataLoader(
        test_event_dataset, batch_size=CFG['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=0 if args.debug else 2,
        drop_last=False
    )

    return {
        'scene': (train_scene_loader, test_scene_loader),
        'event': (train_event_loader, test_event_loader),
    }


# def build_loaders(args):
#     """返回每个任务的 (train_loader, test_loader)"""
#     # ---- scene (MovieNet) ----

#     labels = read_pkl(args.labels)
#     modalA_feat = read_pkl(args.modelA_feat)
#     modalB_feat = read_pkl(args.modelB_feat)
#     seg_sz = args.seg_sz
#     with open(args.split_path, 'r') as f:
#         data = js.load(f)
#         splitSet = data['train']  + data['val']

#     train_scene_dataset = MovieNetDataset(
#         labels, modalA_feat, modalB_feat, splitSet, seg_sz, "train", None
#     )
#     train_scene_loader = torch.utils.data.DataLoader(
#         train_scene_dataset, batch_size=CFG["batch_size"],
#         shuffle=True, drop_last=True, num_workers=0
#     )
#     test_scene_dataset = MovieNetDataset(
#         labels, modalA_feat, modalB_feat, data["test"], seg_sz, "test", None
#     )
#     test_scene_loader = torch.utils.data.DataLoader(
#         test_scene_dataset, batch_size=CFG["batch_size"],
#         shuffle=False, drop_last=False, num_workers=0
#     )

#     # ---- event (KineticsGEBD) ----
#     feature_folder = Path(args.feature_path)
#     score_path = Path(args.score_path)
#     anno_file = Path(args.annotation_path)

#     train_event_dataset = KineticsGEBD(feature_folder, score_path, anno_file, "train", args)
#     train_event_loader = torch.utils.data.DataLoader(
#         train_event_dataset, batch_size=CFG['batch_size'], shuffle=True,
#         collate_fn=collate_fn, num_workers=2, drop_last=True
#     )
#     test_event_dataset = KineticsGEBD(feature_folder, score_path, anno_file, "val", args)
#     test_event_loader = torch.utils.data.DataLoader(
#         test_event_dataset, batch_size=CFG['batch_size'], shuffle=False,
#         collate_fn=collate_fn, num_workers=2, drop_last=False
#     )

#     return {
#         'scene': (train_scene_loader, test_scene_loader),
#         'event': (train_event_loader, test_event_loader),
#     }


# ============================================================
# Loss
# ============================================================
def compute_loss(task_name, score_logit, center, batch, device):
    if task_name == 'scene':
        # batch 是 tuple: (names, img_ctx, plc_ctx, label, pos, ids, ind, _)
        # 这里我们已在 train 里拆开传进来，只需要 label
        label = batch['label'].float().to(device)
        loss = F.binary_cross_entropy_with_logits(score_logit.squeeze(-1), label)
    else:  # event
        cls_target = batch['cls_label'].float().to(device)
        loss = F.binary_cross_entropy_with_logits(score_logit.squeeze(-1), cls_target)
        if center is not None and 'center_label' in batch:
            ct = batch['center_label'].float().to(device)
            pos = cls_target > 0.5
            if pos.sum() > 0:
                loss = loss + F.l1_loss(center[pos], ct[pos])
    return loss



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

# ============================================================
# Trainer
# ============================================================
class ContinualTrainer:
    def __init__(self, args, model, device, loaders):
        self.args = args
        self.model = model
        self.device = device
        self.loaders = loaders
        self.best_metric = None  # 当前任务最优指标
        self.per_task_masks = {}
        self.consolidated_masks = {}

    # -------------------- 训练一个 epoch --------------------
    def train_one_epoch(self, train_loader, optimizer, task_name):
        self.model.train()
        total_loss, n = 0.0, 0
        for batch in tqdm(train_loader, desc=f"Train({task_name})"):
            x, batch_dict = self._unpack_batch(batch, task_name)
            x = x.to(self.device)
            optimizer.zero_grad()
            feature, (score_logit, center) = self.model(
                x, task_name, mask=None, mode='train'
            )
            loss = compute_loss(task_name, score_logit, center, batch_dict, self.device)
            loss.backward()

            # 屏蔽已用参数梯度
            self._mask_gradients()

            optimizer.step()
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
        return total_loss / max(n, 1)

    def _mask_gradients(self):
        if not self.consolidated_masks:
            return
        for key, m in self.consolidated_masks.items():
            if 'last' in key:
                continue
            parts = key.split('.')
            obj = self.model
            ok = True
            for p in parts[:-1]:
                if not hasattr(obj, p):
                    ok = False
                    break
                obj = getattr(obj, p)
            if not ok:
                continue
            attr = parts[-1]
            param = getattr(obj, attr, None)
            if param is not None and param.grad is not None:
                param.grad[m == 1] = 0

    def _unpack_batch(self, batch, task_name):
        """
        把不同任务的 batch 统一成 (x, dict)
        ⚠️ 字段名按你 dataset 实际返回调整
        """
        if task_name == 'scene':
            names, img_ctx, plc_ctx, label, pos, ids, ind = batch
            w = 0.7
            x = w * img_ctx + (1 - w) * plc_ctx
            return x, {'label': label}
        else:  # event
            _, features, targets_list, _, _, _ = batch
            cls_label, center_gt, has_pos, _ = build_center_reg_labels_from_relative_targets(targets_list)
            cls_label = cls_label   # (B,)
            center_gt = center_gt   # (B,)
            has_pos = has_pos       # (B,)
            # 从 targets_list 里抽出训练所需的标签
            d = {'cls_label': cls_label}
            if center_gt is not None:
                d['center_label'] = center_gt
            return features, d

    # -------------------- 验证 --------------------
    @torch.no_grad()
    def validate(self, loader, task_name, ep=0, mask=None):
        self.model.eval()
        if task_name == "event":
            results = self.test_epoch_event(testload=loader, task_name=task_name, mask=mask)
            print(f"✅ Collected predictions for {len(results)} videos")

            eval_res = evaluate_multi_thresholds(
                results,
                anno_root_path=str(CFG['kinetics_dataset_path'] + CFG['kinetics_data']),
                thresholds=[0.10, 0.20, 0.30, 0.40, 0.50],
                rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
                step_frames=1,
                log_tag='event_eval',
            )
            event_eval_record = {
                "stage": f"event_eval_ep{ep}",
                "epoch": ep,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_videos": len(results),
                "results": eval_res,
            }
            save_json(
                event_eval_record,
                os.path.join(CFG["save_path"], f"event_eval_ep{ep}.json")
            )
            return eval_res

        elif task_name == "scene":
            met, moviePL = self.eval_epoch_scene(test_loader=loader, task_name=task_name, mask=mask)
            print(f"[Epoch {ep}] metric = {met}")
            scene_eval_record = {
                "stage": f"scene_eval_ep{ep}",
                "epoch": ep,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metric": met,
                "moviePL": moviePL,
            }
            save_json(
                scene_eval_record,
                os.path.join(CFG["save_path"], f"scene_eval_ep{ep}.json")
            )
            return met

    @torch.no_grad()
    def eval_epoch_scene(self, test_loader, task_name, mask=None, device='cuda:0'):
        self.model.eval()
        predlist, labelist, pathlist, idlist = [], [], [], []
        for sample in tqdm(test_loader, desc="Eval(Scene)"):
            names, img_ctx, plc_ctx, label, pos, ids, ind = sample
            img_ctx = img_ctx.to(device, non_blocking=True)
            plc_ctx = plc_ctx.to(device, non_blocking=True)
            w = 0.7
            x = w * img_ctx + (1 - w) * plc_ctx

            _, (logits, center) = self.model(x, task_name, mask=mask, mode='test')
            probs = center.squeeze().cpu().numpy()

            predlist.append(probs)
            labelist.append(label.numpy())
            pathlist.append(names)
            idlist.append(np.asarray(ind))
        met, moviePL = metric(pathlist, predlist, labelist)
        return met, moviePL

    @torch.no_grad()
    def test_epoch_event(self, testload, task_name, mask=None, gpu=0):
        self.model.eval().cuda(gpu)
        results_dict = defaultdict(list)

        for batch in tqdm(testload, desc="Eval(Event)"):
            locations, features, targets_list, num_frames, base, coherence = batch
            B, L, _ = features.shape
            features = features.cuda(gpu, non_blocking=True)

            _, (score_logit, center_pred) = self.model(features, task_name, mask=mask, mode='test')
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
                    "boundaries": abs_frames[i:i + 1].detach().cpu(),
                    "scores": probs[i:i + 1].detach().cpu(),
                })
        return dict(results_dict)

    # -------------------- 取指标的标量 --------------------
    def _extract_score(self, met, task_name):
        """从评估结果里取一个标量做 best 比较"""
        if task_name == 'scene':
            if isinstance(met, dict):
                key = next(iter(met.keys()))
                return float(met[key])
            return float(met)
        else:  # event
            # 假设 evaluate_multi_thresholds 返回里有 'avg_f1' 字段，按你实际改
            if isinstance(met, dict):
                # 找一个综合指标
                for k in ['avg_f1', 'f1', 'mAP']:
                    if k in met:
                        return float(met[k])
                # 退化为字典里第一个数值
                for v in met.values():
                    if isinstance(v, (int, float)):
                        return float(v)
            return 0.0

    # -------------------- 训练一个 task --------------------
    def train_task(self, task_id, task_name):
        train_loader, test_loader = self.loaders[task_name]
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=0.9)
        # # ---------- 改用 AdamW 优化器 ----------
        # optimizer = optim.AdamW(
        #     self.model.parameters(),
        #     lr=self.args.lr,
        #     weight_decay=1e-4,          # AdamW 的 weight decay
        #     betas=(0.9, 0.999)
        # )

        # # ---------- 添加 CosineAnnealing 调度器 ----------
        # total_epochs = self.args.n_epochs
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=total_epochs, eta_min=1e-6
        # )

        save_dir = CFG["save_path"]
        os.makedirs(save_dir, exist_ok=True)
        best_path = os.path.join(save_dir, f"{task_name}_best.pth")
        self.best_metric = None  # 重置

        history = []

        for epoch in range(1, self.args.n_epochs + 1):
            tr_loss = self.train_one_epoch(train_loader, optimizer, task_name)
            print(f"[Task {task_id}/{task_name}] Epoch {epoch} train_loss={tr_loss:.4f}")

            # 验证 (用当前训练中状态，不用 mask，看 raw 性能)
            met = self.validate(test_loader, task_name, ep=epoch, mask=None)

            # 保存每个 epoch 的 ckpt
            ckpt = {
                "model": self.model.state_dict(),
                "epoch": epoch,
                "task_id": task_id,
                "task_name": task_name,
                "consolidated_masks": self.consolidated_masks,
                "per_task_masks": self.per_task_masks,
            }
            torch.save(ckpt, os.path.join(save_dir, f'{task_name}_ep{epoch}.pth'))

            # 取分数比较 best
            score = self._extract_score(met, task_name)
            history.append({"epoch": epoch, "score": score, "loss": tr_loss})

            if (self.best_metric is None) or (score > self.best_metric):
                self.best_metric = score
                torch.save(ckpt, best_path)
                print(f"  >> New best ({task_name}={score:.4f}) saved -> {best_path}")

        # # ---------- 每个 epoch 后更新学习率 ----------
        # scheduler.step()
        # 训练完该任务，加载 best 权重
        print(f"\n[Task {task_id}/{task_name}] Loading best ckpt: {best_path}")
        best_ckpt = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(best_ckpt["model"])

        # 保存当前任务 mask
        self.per_task_masks[task_id] = self.model.get_masks(task_id)

        # 合并 mask
        if task_id == 0:
            self.consolidated_masks = deepcopy(self.per_task_masks[0])
        else:
            for key in self.per_task_masks[task_id]:
                a = self.consolidated_masks.get(key)
                b = self.per_task_masks[task_id][key]
                if a is not None and b is not None:
                    self.consolidated_masks[key] = 1 - ((1 - a) * (1 - b))
                elif b is not None:
                    self.consolidated_masks[key] = b

        # 打印参数占用
        if self.consolidated_masks:
            total_params = sum(m.numel() for m in self.consolidated_masks.values())
            used_params = sum(m.sum().item() for m in self.consolidated_masks.values())
            print(f"  >> Consolidated mask usage: "
                  f"{used_params/total_params*100:.2f}% "
                  f"({int(used_params)}/{total_params})")

        # 保存训练历史
        save_json(
            {"task_id": task_id, "task_name": task_name, "history": history,
             "best_metric": self.best_metric},
            os.path.join(save_dir, f"{task_name}_history.json")
        )

        return self.best_metric


# ============================================================
# 主流程
# ============================================================
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    # ⭐ 调试模式：覆盖一些参数
    if args.debug:
        args.n_epochs = args.debug_epochs
        args.batch_size = min(args.batch_size, 8)
        CFG["batch_size"] = args.batch_size
        print(f"🐛 DEBUG MODE: epochs={args.n_epochs}, bs={args.batch_size}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(CFG["save_path"], exist_ok=True)

    # 数据
    loaders = build_loaders(args)

    # 模型
    model = LocalUncertaintyAwareGraphAttentionLite(
        dim_in=2048,
        num_heads=8,
        window_size=9,
        sparsity=args.sparsity,
        size=args.size,
    ).to(device)

    # Trainer
    trainer = ContinualTrainer(args, model, device, loaders)

    # ========== 顺序训练 ==========
    task_best_scores = {}
    for task_id, task_name in enumerate(TASKS):
        print(f"\n{'#'*60}\n#  Task {task_id}: {task_name.upper()}\n{'#'*60}")
        best_score = trainer.train_task(task_id, task_name)
        task_best_scores[task_name] = best_score

    # ========== 最终联合评测 ==========
    print(f"\n{'='*60}\n  FINAL EVALUATION ON ALL TASKS\n{'='*60}")
    final_results = {}
    for task_id, task_name in enumerate(TASKS):
        print(f"\n--- Final eval: {task_name} ---")
        _, test_loader = loaders[task_name]
        # 用该任务训练时保存的 mask 测
        mask = trainer.per_task_masks[task_id]
        met = trainer.validate(test_loader, task_name, ep=999, mask=mask)
        final_results[task_name] = {
            "metric": met,
            "score": trainer._extract_score(met, task_name),
            "best_during_training": task_best_scores[task_name],
        }

    # 汇总保存
    summary = {
        "tasks": TASKS,
        "args": vars(args),
        "final_results": final_results,
        "task_best_during_training": task_best_scores,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(summary, os.path.join(CFG["save_path"], "final_summary.json"))

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for tn, res in final_results.items():
        print(f"  [{tn:6s}]  final_score = {res['score']:.4f}   "
              f"(best during training: {res['best_during_training']:.4f})")
    print(f"\nAll results saved to: {CFG['save_path']}")


# ============================================================
# CLI
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--sparsity', type=float, default=0.5)
    parser.add_argument('--size', type=int, default=21)

    # ⭐ 新增：调试模式
    parser.add_argument('--debug', action='store_true',
                        help='快速跑通模式：只用很少数据')
    parser.add_argument('--debug_train_samples', type=int, default=64,
                        help='debug 模式下，每个任务训练用多少样本')
    parser.add_argument('--debug_test_samples', type=int, default=32,
                        help='debug 模式下，每个任务测试用多少样本')
    parser.add_argument('--debug_epochs', type=int, default=1)
    # parser.add_argument('--feature_path', type=str, default='data/Kinetics/features')
    # parser.add_argument('--score_path', type=str, default='data/Kinetics/scores')
    # parser.add_argument('--annotation_path', type=str, default='data/Kinetics/anno.json')
    # parser.add_argument('--movienet_path', type=str, default='data/Movienet')

    parser.add_argument('--save_path', type=str, default='<save_dir>')
    args = parser.parse_args()

    CFG["save_path"] = args.save_path
    CFG["batch_size"] = args.batch_size

    feature_path = CFG["kinetics_dataset_path"] + "features"
    annotation_path = CFG["kinetics_dataset_path"] + "data"
    score_path = CFG["kinetics_dataset_path"] + "data"
    args.feature_path = feature_path
    args.annotation_path = annotation_path
    args.score_path = score_path
    args.window_size = CFG["T"]
    args.interval = 1
    args.device = "cuda:0"

    args.modelA_feat =CFG["movienet_dataset_path"] + CFG["modalA_path"] 
    args.modelB_feat =CFG["movienet_dataset_path"] + CFG["modalB_path"] 
    args.seg_sz = CFG["seg_sz"]
    args.labels = CFG["movienet_dataset_path"] + CFG["label_path"]
    args.split_path = CFG["movienet_dataset_path"] + CFG["split_path"]

    
    print('=' * 60)
    print('Arguments:')
    for k, v in vars(args).items():
        print(f'  {k:20s}: {v}')
    print('=' * 60)

    main(args)
