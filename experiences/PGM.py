# main_continual.py
"""
持续学习主入口：
- 任务1: scene boundary detection (MovieNet)
- 任务2: event boundary detection (KineticsGEBD)
"""
import os
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
from model.backbone.subnet import SubnetLinear
from model.backbone.RelationNet_PGM import LocalUncertaintyAwareGraphAttentionLite

# ====================================
# ⚠️ 这两个 import 替换成你自己的实现
# ====================================
from dataset.movienet.load_movienet_server import MovieNetDataset, read_pkl, read_pkl2
from dataset.kinetics.dataset_became import KineticsGEBD, collate_fn
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

    # 两任务的类别数（用于建立两个分类头）
    'task1_num_classes': 10,
    'task2_num_classes': 10,

    # epoch 设置
    'task1_epochs':        5,   # Task1 总训练轮数
    'task2_stage1_epochs': 5, # Task2 稳定容器阶段（SVD 投影）
    'task2_stage2_epochs': 5, # Task2 可塑性 + 蒸馏 + 融合阶段

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
    'task_order': ['event', 'scene'],   # 或 ['scene', 'event']
    'save_path': '/root/autodl-fs/txx_code/became/event'
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--order', nargs=2, choices=['event', 'scene'],
                   default=CFG['task_order'],
                   help="例：--order event scene 或 --order scene event")
    return p.parse_args()

# 默认任务顺序
TASKS = ['scene', 'event']  # 可改成 ['event', 'scene'] 做对比实验


# ============================================================
#   1. 数据加载（按你已有代码封装）
# ============================================================
def build_loaders(args, CFG):
    """
    返回: { 'scene': (train_loader, test_loader),
            'event': (train_loader, test_loader) }
    """
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
        data = json.load(f)
        splitSet = data['train']  + data['val']

    modalA_feat = read_pkl2(CFG["movienet_dataset_path"] + CFG["modalA_path"])
    modalB_feat = read_pkl2(CFG["movienet_dataset_path"] + CFG["modalB_path"])
    seg_sz = CFG["seg_sz"]
    labels = read_pkl(CFG["movienet_dataset_path"] + CFG["label_path"])
    # ---- scene (MovieNet) ----
    # 你需要自己提供以下变量：
    #   labels, modalA_feat, modalB_feat, splitSet, data, seg_sz
    train_scene_dataset = MovieNetDataset(
        labels, modalA_feat, modalB_feat, splitSet, seg_sz, "train", None
    )
    train_scene_loader = torch.utils.data.DataLoader(
        train_scene_dataset, batch_size=CFG["batch_size"],
        shuffle=True, drop_last=True, num_workers=0
    )
    test_scene_dataset = MovieNetDataset(
        labels, modalA_feat, modalB_feat, data["test"], seg_sz, "test", None
    )
    test_scene_loader = torch.utils.data.DataLoader(
        test_scene_dataset, batch_size=CFG["batch_size"],
        shuffle=False, drop_last=False, num_workers=0
    )

    # ---- event (KineticsGEBD) ----
    feature_folder = Path(args.feature_path)
    score_path = Path(args.score_path)
    anno_file = Path(args.annotation_path)

    train_event_dataset = KineticsGEBD(feature_folder, score_path, anno_file, "train", args)
    train_event_loader = torch.utils.data.DataLoader(
        train_event_dataset, batch_size=CFG['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=False, drop_last=True
    )
    test_event_dataset = KineticsGEBD(feature_folder, score_path, anno_file, "val", args)
    test_event_loader = torch.utils.data.DataLoader(
        test_event_dataset, batch_size=CFG['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=False, drop_last=False
    )

    return {
        'scene': (train_scene_loader, test_scene_loader),
        'event': (train_event_loader, test_event_loader),
    }

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
#   2. Loss 计算（每个任务不同）
# ============================================================
def compute_loss(task_name, score_logit, center, batch, device):
    """
    ⚠️ 字段名（'feature', 'label' 等）请按你的 Dataset.__getitem__ 返回为准
    """
    if task_name == 'scene':
        _, img_s, plc_s, label_s, pos, _, _ = batch
        label_s = label_s.to(device).float()
        
        # MlpHead 输出已经过 sigmoid（看你的代码 x = self.act(x1)）
        # 如果 score_logit 是过 sigmoid 后的，就用 BCE；如果是原始 logit，就用 BCEWithLogits
        # 我这里假设 score_logit 是原始 logit
        loss = F.binary_cross_entropy_with_logits(score_logit.squeeze(-1), label_s)
    else:  # event
        _, features, targets_list, _, _, _ = batch
        cls_label, center_gt, has_pos, _ = build_center_reg_labels_from_relative_targets(targets_list)
        cls_label = cls_label.to(device)   # (B,)
        center_gt = center_gt.to(device)   # (B,)
        has_pos = has_pos.to(device)       # (B,)

        features = features.to(device)

        loss = F.binary_cross_entropy_with_logits(score_logit.squeeze(-1), cls_label)

        if center is not None and center_gt:
            ct = center_gt.float().to(device)
            pos = cls_label > 0.5
            if pos.sum() > 0:
                loss = loss + F.l1_loss(center[pos], ct[pos])
    return loss


# ============================================================
#   3. 训练一个 task
# ============================================================
def train_one_task(args, model, device, train_loader, optimizer,
                   task_name, consolidated_masks):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in train_loader:
        if task_name == "event":
            _, x, targets_list, _, _, _ =  batch
        else:
            _, img_s, plc_s, label_s, pos, _, _ = batch
            img_s = img_s.to(device)
            plc_s = plc_s.to(device)
            w = 0.7
            x = w * img_s + (1 - w) * plc_s
        x = x.to(device)  # (B, T, 2048)
        optimizer.zero_grad()
        feature, (score_logit, center) = model(
            x, task_name, mask=None, mode='train'
        )
        loss = compute_loss(task_name, score_logit, center, batch, device)
        loss.backward()

        # ⭐ 屏蔽已被先前任务占用的参数梯度
        if consolidated_masks:
            for key, m in consolidated_masks.items():
                if 'last' in key:
                    continue
                # 解析 key 路径（如 'q_proj.weight' / 'a.b.weight'）
                parts = key.split('.')
                obj = model
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

        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)

    return total_loss / max(n, 1)


# ============================================================
#   4. 评估一个 task
# ============================================================
@torch.no_grad()
def evaluate_task(args, model, device, test_loader, task_name, curr_task_masks):
    model.eval()
    total_loss, total_correct, total_num = 0.0, 0, 0

    for batch in test_loader:
        x = batch['feature'].to(device)
        feature, (score_logit, center) = model(
            x, task_name, mask=curr_task_masks, mode='test'
        )
        loss = compute_loss(task_name, score_logit, center, batch, device)
        total_loss += loss.item() * x.size(0)

        target_key = 'label' if task_name == 'scene' else 'cls_label'
        target = batch[target_key].long().to(device)
        pred = (torch.sigmoid(score_logit) > 0.5).long()
        total_correct += (pred == target).sum().item()
        total_num += x.size(0)

    avg_loss = total_loss / max(total_num, 1)
    acc = 100. * total_correct / max(total_num, 1)
    return avg_loss, acc


# ============================================================
#   5. 主流程
# ============================================================
def main(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CFG = {'batch_size': args.batch_size}
    loaders = build_loaders(args, CFG)

    model = LocalUncertaintyAwareGraphAttentionLite(
        dim_in=2048,
        num_heads=8,
        window_size=9,
        sparsity=args.sparsity,
        size=args.size,
    ).to(device)

    print('=' * 60)
    print('Model Parameters:')
    for n, p in model.named_parameters():
        print(f'  {n:60s} {tuple(p.shape)}')
    print('=' * 60)

    n_tasks = len(TASKS)
    acc_matrix = np.zeros((n_tasks, n_tasks))
    per_task_masks, consolidated_masks = {}, {}

    for task_id, task_name in enumerate(TASKS):
        print(f'\n{"#"*60}')
        print(f'#  Task {task_id}: {task_name.upper()}')
        print(f'{"#"*60}')

        train_loader, test_loader = loaders[task_name]
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        # 或者 Adam:
        # optimizer = optim.Adam(model.parameters(), lr=args.lr)

        best_val_acc = 0.0
        for epoch in range(1, args.n_epochs + 1):
            tr_loss = train_one_task(
                args, model, device, train_loader,
                optimizer, task_name, consolidated_masks
            )
            val_loss, val_acc = evaluate_task(
                args, model, device, test_loader, task_name,
                curr_task_masks=None
            )
            print(f'  [Task {task_id}/{task_name}] Epoch {epoch:3d} | '
                  f'tr_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.2f}%')

            if val_acc > best_val_acc:
                best_val_acc = val_acc

        # ⭐ 保存当前任务的 mask
        per_task_masks[task_id] = model.get_masks(task_id)

        # ⭐ 合并 mask（OR 操作）
        if task_id == 0:
            consolidated_masks = deepcopy(per_task_masks[0])
        else:
            for key in per_task_masks[task_id]:
                a = consolidated_masks.get(key)
                b = per_task_masks[task_id][key]
                if a is not None and b is not None:
                    consolidated_masks[key] = 1 - ((1 - a) * (1 - b))
                elif b is not None:
                    consolidated_masks[key] = b

        # 打印 sparsity
        total_params, used_params = 0, 0
        for k, m in consolidated_masks.items():
            total_params += m.numel()
            used_params += m.sum().item()
        print(f'  >> Consolidated mask usage: '
              f'{used_params/total_params*100:.2f}% '
              f'({int(used_params)}/{total_params})')

        # ⭐ 测所有学过/未学的任务（构造 acc 矩阵）
        for jj, prev_task in enumerate(TASKS):
            _, prev_test_loader = loaders[prev_task]
            mask = per_task_masks[jj] if jj <= task_id else per_task_masks[task_id]
            _, acc_matrix[task_id, jj] = evaluate_task(
                args, model, device, prev_test_loader, prev_task,
                curr_task_masks=mask
            )

        print(f'\n  Acc Matrix after Task {task_id}:')
        for i_a in range(task_id + 1):
            row = '    '
            for j_a in range(n_tasks):
                row += f'{acc_matrix[i_a, j_a]:6.2f} '
            print(row)

    # ============================================================
    # 最终统计
    # ============================================================
    print('\n' + '=' * 60)
    print('Final Results')
    print('=' * 60)

    # 对角线：训练任务自身性能
    diag_avg = np.mean([acc_matrix[i, i] for i in range(n_tasks)])
    # 最后一行：所有任务训完后，每个任务的最终性能
    final_avg = np.mean(acc_matrix[-1])
    # BWT (Backward Transfer)
    if n_tasks > 1:
        bwt = np.mean(acc_matrix[-1, :-1] - np.diag(acc_matrix)[:-1])
    else:
        bwt = 0.0

    print(f'Task Order      : {TASKS}')
    print(f'Diagonal AvgAcc : {diag_avg:.2f}%')
    print(f'Final  AvgAcc   : {final_avg:.2f}%')
    print(f'Backward Transfer: {bwt:.2f}%')

    # 保存结果
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.name)
    np.save(save_path + '_acc_matrix.npy', acc_matrix)
    torch.save({
        'model_state': model.state_dict(),
        'per_task_masks': per_task_masks,
        'consolidated_masks': consolidated_masks,
        'acc_matrix': acc_matrix,
        'tasks': TASKS,
    }, save_path + '_ckpt.pth')
    print(f'\nSaved to {save_path}_*.npy / .pth')


# ============================================================
#   6. 命令行参数
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Continual: Scene + Event Boundary Detection')

    # 训练
    parser.add_argument('--n_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=1)

    # 模型 / 持续学习
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='每层保留的参数比例（每个任务可用预算）')
    parser.add_argument('--size', type=int, default=21,
                        help='序列长度 T')

    # 数据路径（event）
    # parser.add_argument('--feature_path', type=str, default='./data/gebd/features')
    # parser.add_argument('--score_path', type=str, default='./data/gebd/scores')
    # parser.add_argument('--annotation_path', type=str, default='./data/gebd/anno.json')

    # 数据路径（scene）—— 你自己的 MovieNet 路径
    # parser.add_argument('--movienet_path', type=str, default='./data/movienet')

    # 保存
    parser.add_argument('--save_dir', type=str, default='/root/autodl-fs/pgm/results')
    parser.add_argument('--name', type=str, default='cl_scene_event')

    args = parser.parse_args()

    print('=' * 60)
    print('Arguments:')
    for k, v in vars(args).items():
        print(f'  {k:20s}: {v}')
    print('=' * 60)

    main(args)