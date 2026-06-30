"""
Event-First 三阶段独立训练脚本 (event → scene → event)，
镜像原 SceneFirst_WithCL_AdvOnly 的完整 SchemeA：
    KD (特征蒸馏) + GRL 对抗 + 每 step 更新一次全局 Disc

用法：
    python train_3stages_event_first.py --stage 1   # Event-Only (from scratch)
    python train_3stages_event_first.py --stage 2   # Scene + Event-Teacher KD + Adv
    python train_3stages_event_first.py --stage 3   # 冻结 backbone + 新 FrameWiseHead

每阶段一个独立进程，结束即退出，OS 回收 RAM / 显存 / DataLoader workers。
阶段间通过磁盘 ckpt 衔接：
    Stage 1 -> CKPT_DIR/event_only_best.pth
    Stage 2 -> CKPT_DIR/scene_with_event_teacher_best.pth
    Stage 3 -> CKPT_DIR/event_head_best.pth
"""

import argparse
import os
import sys
from typing import Optional, Tuple
from collections import defaultdict
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

# ======== 根据你的工程结构调整这些 import ========
from model.backbone.RelationNet import LocalUncertaintyAwareGraphAttentionLite
from model.detector.scene_detector import MlpHead
from model.detector.event_detector import FrameWiseHead
from model.discriminator.discriminator import TemporalDiscriminator
from dataset.movienet.load_movienet_server2 import load_data
from dataset.kinetics.dataset import build as build_kinetics, collate_fn
from tool.warmup_lr import warmup_decay_cosine
from tool.metric.scene_metric import metric
from tool.metric.kinetics_metric import (
    save_submission_in_seconds_multi_thresh,
    eval_by_frame_metric_from_seconds_map,
    summarize_global_avg_over_thresholds,
)
# =================================================


# =========================
# Config
# =========================
EVENT_PRETRAIN_EPOCHS = 5    # Stage 1: event 从零
SCENE_TRAIN_EPOCHS    = 5    # Stage 2: scene + event teacher KD + adv
EVENT_HEAD_EPOCHS     = 20   # Stage 3: 冻结 backbone, 新 FrameWiseHead
BATCH_SIZE = 128
T = 21
GPU = 0
DEVICE = f"cuda:{GPU}"


movienet_dataset_path = "data/MovieNet/"
kinetics_dataset_path = "data/Kinetics/"
IMG_PATH         = movienet_dataset_path + 'ImageNet_shot.pkl'
PLC_PATH         = movienet_dataset_path + 'Places_shot.pkl'
LABEL_PATH       = movienet_dataset_path + 'label_endShot.pkl'
SPLIT_PATH_SCENE = movienet_dataset_path + 'split318.json'
MAMBA_PATH       = IMG_PATH

# 输出目录
ck_path = "<weight>"
CKPT_DIR = ck_path + '<checkpoint.pt>'
os.makedirs(CKPT_DIR, exist_ok=True)

# Stage 2 蒸馏 + 对抗超参
ALPHA_FEAT       = 0.5
LAMBDA_ADV       = 0.1
EVENT_KEEP_RATIO = 1.0    # Stage 1 主任务: 事件抽样比例
SCENE_KEEP_RATIO = 1.0    # Stage 2 主任务: 场景抽样比例
EVENT_AUX_RATIO  = 0.30   # Stage 2 辅助: 事件仅供域对抗
USE_PROJ_TO_TEACHER = False

# 三阶段 ckpt 衔接路径
EVENT_ONLY_BEST_PATH               = os.path.join(CKPT_DIR, "event_only_best.pth")
SCENE_WITH_EVENT_TEACHER_BEST_PATH = os.path.join(CKPT_DIR, "scene_with_event_teacher_best.pth")
EVENT_HEAD_BEST_PATH               = os.path.join(CKPT_DIR, "event_head_best.pth")


# =========================
# 工具函数
# =========================
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
    else:
        return obj


def save_json(data, json_path):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, ensure_ascii=False, indent=2)


def build_center_reg_labels_from_relative_targets(targets_list):
    B = len(targets_list)
    device = (targets_list[0]['boundaries'].device
              if torch.is_tensor(targets_list[0]['boundaries'])
              else torch.device('cpu'))

    cls_label = torch.zeros(B, dtype=torch.float32, device=device)
    center_gt = torch.full((B,), 0.5, dtype=torch.float32, device=device)
    has_pos   = torch.zeros(B, dtype=torch.bool, device=device)
    vid_ids   = []

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


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
    return ckpt


def strip_module_prefix(sd: dict):
    return {
        (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in sd.items()
    }


def load_backbone_only(model: nn.Module, ckpt_path: str, detector_prefix: str = "detector."):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = extract_state_dict(ckpt)
    sd = strip_module_prefix(sd)
    pruned = {k: v for k, v in sd.items() if not k.startswith(detector_prefix)}
    load_info = model.load_state_dict(pruned, strict=False)
    print("Backbone-only load -> Missing keys:", load_info.missing_keys)
    print("Backbone-only load -> Unexpected keys:", load_info.unexpected_keys)
    print("Loaded params:", len(pruned))
    return load_info


def require_ckpt(path: str, stage_name: str):
    if not os.path.isfile(path):
        print(f"[ERROR] Stage {stage_name} requires checkpoint: {path}")
        print(f"        请先运行更早阶段, 确认产生该文件后再继续。")
        sys.exit(1)


# =========================
# Hook
# =========================
class EncoderPreHook:
    def __init__(self, model_with_head: nn.Module):
        self.z = None
        def pre_hook(module, inputs):
            z = inputs[0]
            self.z = z.mean(dim=1) if z.dim() == 3 else z
        self.h = model_with_head.detector.register_forward_pre_hook(lambda m, inp: pre_hook(m, inp))

    def close(self):
        try:
            self.h.remove()
        except Exception:
            pass


# =========================
# GRL
# =========================
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


# =========================
# 模型构建
# =========================
def build_encoder(device=DEVICE, dim_in=2048, num_heads=8, window_size=5, size=21):
    return LocalUncertaintyAwareGraphAttentionLite(
        dim_in=dim_in,
        num_heads=num_heads,
        window_size=window_size,
        sim_temperature=0.07,
        neighbor_temp=0.5,
        uncertainty_mode="variance",
        use_relative_pos_bias=True,
        norm="ln",
        size=size,
    ).to(device)


def build_student_with_event_head(device=DEVICE, size=T):
    """Stage 1: encoder + FrameWiseHead, 从头训事件。"""
    student = build_encoder(device=device, size=size)
    student.detector = FrameWiseHead(in_features=2176).to(device)
    hook = EncoderPreHook(student)
    return student, hook, 2176


def build_student_with_scene_head_from_event_ckpt(event_ckpt_path, device=DEVICE,
                                                  size=T, mlp_hid=512, mlp_out=1):
    """Stage 2: encoder ← 最佳事件 ckpt; detector = MlpHead (随机)。"""
    student = build_encoder(device=device, size=size)
    student.detector = MlpHead(in_dim=2176, hid_dim=mlp_hid, out_dim=mlp_out).to(device)
    load_backbone_only(student, event_ckpt_path, detector_prefix="detector.")
    print("✅ Student backbone initialized from best event ckpt; detector = MlpHead")
    hook = EncoderPreHook(student)
    return student, hook, 2176


def build_teacher_encoder_from_event_ckpt(ckpt_path, device=DEVICE, size=T):
    """Stage 2: teacher encoder ← 最佳事件 ckpt (frozen)。"""
    teacher = build_encoder(device=device, size=size)
    load_backbone_only(teacher, ckpt_path, detector_prefix="detector.")
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    teacher.detector = nn.Identity().to(device)
    hook = EncoderPreHook(teacher)
    return teacher, hook, 2176


def build_teacher_event_detector_from_ckpt(ckpt_path, device=DEVICE, in_features=2176):
    """从最佳事件 ckpt 提取 FrameWiseHead 作为冻结的事件检测头 (cross-task 验证)。"""
    head = FrameWiseHead(in_features=in_features).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = extract_state_dict(ckpt)
    sd = strip_module_prefix(sd)
    head_sd = {
        k.replace("detector.", "", 1): v
        for k, v in sd.items() if k.startswith("detector.")
    }
    info = head.load_state_dict(head_sd, strict=False)
    print("Teacher event detector load -> Missing:", info.missing_keys)
    print("Teacher event detector load -> Unexpected:", info.unexpected_keys)
    print("Loaded event-head params:", len(head_sd))
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


# =========================
# Data helpers
# =========================
def build_subset_loader_scene(split_json, keep_ratio, batch_size, seg_sz, mode='train'):
    full_loader = load_data(
        LABEL_PATH, IMG_PATH, PLC_PATH, split_json,
        batch_size, seg_sz=seg_sz, mode1=mode
    )
    dataset = full_loader.dataset
    N = len(dataset)
    if keep_ratio >= 1.0:
        return full_loader, N, N
    keep = max(1, int(N * keep_ratio))
    indices = torch.randperm(N)[:keep].tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=0,
        pin_memory=False, drop_last=True,
        persistent_workers=False,
    )
    return loader, keep, N


def build_subset_loader_event(keep_ratio, batch_size, seg_sz, device=DEVICE):
    args = argparse.Namespace()
    args.feature_path    = kinetics_dataset_path + "features"
    args.score_path      = kinetics_dataset_path + "data"
    args.annotation_path = kinetics_dataset_path + "data"
    args.window_size     = seg_sz
    args.interval        = 1
    args.device          = device

    dataset_full = build_kinetics(split='train', args=args)
    N = len(dataset_full)
    if keep_ratio >= 1.0:
        loader = torch.utils.data.DataLoader(
            dataset_full, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
            pin_memory=False, drop_last=True,
            persistent_workers=False,
        )
        print(f"[Event (Kinetics)] full: {N}")
        return loader, N, N

    keep = max(1, int(N * keep_ratio))
    indices = torch.randperm(N)[:keep].tolist()
    subset = torch.utils.data.Subset(dataset_full, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
        pin_memory=False, drop_last=True,
        persistent_workers=False,
    )
    print(f"[Event (Kinetics)] subset: {keep}/{N} ({keep/N:.1%})")
    return loader, keep, N


def paired_iter(loader_a, loader_b):
    ita, itb = iter(loader_a), iter(loader_b)
    while True:
        try:
            ba = next(ita)
        except StopIteration:
            ita = iter(loader_a)
            ba = next(ita)
        try:
            bb = next(itb)
        except StopIteration:
            itb = iter(loader_b)
            bb = next(itb)
        yield ba, bb


# =========================
# STAGE 1: 事件独立预训练
# =========================
def train_epoch_event_only(train_loader_event, student, opti, lr_sh=None, device=DEVICE):
    student.train().to(device)
    pbar = tqdm(train_loader_event, desc="Stage 1: Train(Event only)")

    for batch in pbar:
        locations, features, targets_list, num_frames, base, coherence = batch
        features = features.to(device, non_blocking=True)
        cls_label, center_gt, has_pos, _ = build_center_reg_labels_from_relative_targets(targets_list)
        cls_label = cls_label.to(device, non_blocking=True).float()

        opti.zero_grad(set_to_none=True)
        score_logit, center_pred = student(features)
        loss = F.binary_cross_entropy_with_logits(score_logit, cls_label)
        loss.backward()
        opti.step()
        if lr_sh is not None:
            lr_sh.step()
        pbar.set_postfix(loss=float(loss.item()))


# =========================
# STAGE 2: 场景训练 + 事件 teacher KD + 域对抗 (完整 SchemeA)
#   - 主任务: 场景 BCE   (新域 = 场景)
#   - 辅助:  事件        (旧域, 仅供域对抗)
#   - Teacher: 来自最佳事件 ckpt 的 encoder (frozen)
#   - 每 step 都更新 Disc, 然后 GRL 对学生反向
# =========================
def train_epoch_scene_with_event_teacher(
    train_loader_scene: DataLoader,        # 主任务 (新域 = 场景)
    train_loader_event_aux: DataLoader,    # 旧域 (事件), 仅供域对抗
    student, stu_hook, stu_z_dim,
    teacher, tea_hook, tea_z_dim,
    disc,
    opt_main, opt_disc,
    lr_sh_main=None, lr_sh_disc=None,
    device=DEVICE,
    alpha_feat=ALPHA_FEAT,
    lambda_adv=LAMBDA_ADV,
    proj_to_teacher: bool = USE_PROJ_TO_TEACHER,
):
    student.train().to(device)
    teacher.eval().to(device)
    disc.train().to(device)

    proj = nn.Identity().to(device)
    if proj_to_teacher or (stu_z_dim != tea_z_dim):
        proj = nn.Linear(stu_z_dim, tea_z_dim, bias=False).to(device)

    # 主任务 epoch 长度跟随 scene loader
    it = paired_iter(train_loader_scene, train_loader_event_aux)
    pbar = tqdm(range(len(train_loader_scene)),
                desc="Stage 2: Train(Scene primary + Event aux) + SchemeA")

    for _ in pbar:
        batch_scene, batch_event = next(it)
        (name, img_s, plc_s, label_s, pos, _, _) = batch_scene
        (locations, features, targets_list, num_frames, base, coherence) = batch_event

        # 场景 (主任务/新域)
        img_s   = img_s.to(device, non_blocking=True)
        plc_s   = plc_s.to(device, non_blocking=True)
        label_s = label_s.to(device, non_blocking=True).float()
        w = 0.7
        x_new = w * img_s + (1 - w) * plc_s

        # 事件 (旧域, 仅 adv 用)
        features = features.to(device, non_blocking=True)
        x_old = features

        # 1) 更新域判别器
        opt_disc.zero_grad(set_to_none=True)
        with torch.no_grad():
            _ = student(x_new)
            z_new = stu_hook.z.detach()
            _ = student(x_old)
            z_old = stu_hook.z.detach()
        x_dom = torch.cat([z_old, z_new], dim=0)
        y_dom = torch.cat([
            torch.zeros(z_old.size(0), 1, device=device),
            torch.ones(z_new.size(0), 1, device=device),
        ], dim=0)
        logit_d = disc(x_dom)
        loss_disc = F.binary_cross_entropy_with_logits(logit_d, y_dom)
        loss_disc.backward()
        opt_disc.step()
        if lr_sh_disc is not None:
            lr_sh_disc.step()

        # 2) 更新学生
        opt_main.zero_grad(set_to_none=True)

        # 主任务: 场景 BCE
        logits, _center = student(x_new)
        logits = logits.squeeze(-1)
        loss_task = F.binary_cross_entropy_with_logits(logits, label_s)

        # 特征蒸馏: 在新域(scene)输入上, 让 student encoder 仍贴近 event-pretrained teacher
        with torch.no_grad():
            _ = teacher(x_new)
            z_t_new = tea_hook.z
        _ = student(x_new)
        z_s_new = stu_hook.z
        z_s_new_proj = proj(z_s_new)
        loss_feat = F.mse_loss(z_s_new_proj, z_t_new)

        # 域对抗
        _ = student(x_old)
        z_s_old = stu_hook.z
        z_all = torch.cat([z_s_old, z_s_new], dim=0)
        y_all = torch.cat([
            torch.zeros(z_s_old.size(0), 1, device=device),
            torch.ones(z_s_new.size(0), 1, device=device),
        ], dim=0)
        logits_adv = disc(grad_reverse(z_all, lambd=1.0))
        loss_adv = F.binary_cross_entropy_with_logits(logits_adv, y_all)

        loss_total = loss_task + alpha_feat * loss_feat + lambda_adv * loss_adv
        loss_total.backward()
        opt_main.step()
        if lr_sh_main is not None:
            lr_sh_main.step()

        pbar.set_postfix(
            task=float(loss_task.item()),
            feat=float(loss_feat.item()),
            adv=float(loss_adv.item()),
            d=float(loss_disc.item()),
            total=float(loss_total.item()),
        )


# =========================
# STAGE 3: 冻结 backbone, 只训新的 FrameWiseHead
# =========================
def train_epoch_event_head_only(train_loader, model, opti, lr_sh=None, device=DEVICE):
    model.train().to(device)
    pbar = tqdm(train_loader, desc="Stage 3: Train(Event head only)")
    for batch in pbar:
        locations, features, targets_list, num_frames, base, coherence = batch
        features = features.to(device, non_blocking=True)
        cls_label, center_gt, has_pos, _ = build_center_reg_labels_from_relative_targets(targets_list)
        cls_label = cls_label.to(device, non_blocking=True).float()

        opti.zero_grad(set_to_none=True)
        score_logit, center_pred = model(features)
        loss = F.binary_cross_entropy_with_logits(score_logit, cls_label)
        loss.backward()
        opti.step()
        if lr_sh is not None:
            lr_sh.step()
        pbar.set_postfix(loss=float(loss.item()))


# =========================
# Eval
# =========================
@torch.no_grad()
def eval_epoch_scene(test_loader, model, device=DEVICE):
    model.eval()
    predlist, labelist, pathlist, idlist = [], [], [], []
    for sample in tqdm(test_loader, desc="Eval(Scene)"):
        names, img_ctx, plc_ctx, label, pos, ids, ind = sample
        img_ctx = img_ctx.to(device, non_blocking=True)
        plc_ctx = plc_ctx.to(device, non_blocking=True)
        w = 0.7
        x = w * img_ctx + (1 - w) * plc_ctx
        logits, center = model(x)
        probs = center.squeeze().cpu().numpy()
        predlist.append(probs)
        labelist.append(label.numpy())
        pathlist.append(names)
        idlist.append(np.asarray(ind))
    met, moviePL = metric(pathlist, predlist, labelist)
    return met, moviePL


@torch.no_grad()
def test_epoch_event(testload, model, gpu=0):
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
                "boundaries": abs_frames[i:i+1].detach().cpu(),
                "scores": probs[i:i+1].detach().cpu(),
            })
    return dict(results_dict)


def evaluate_multi_thresholds(
    results,
    anno_root_path,
    thresholds=None,
    rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
    step_frames=1,
    log_tag="event_eval",
):
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.30, 0.40, 0.50]
    seconds_map_by_thresh = save_submission_in_seconds_multi_thresh(
        results, anno_root_path, thresholds,
        step_frames=step_frames,
        out_dir=f"{log_tag}_pred_outputs",
        write_pkls=False, debug=True,
    )
    eval_results = eval_by_frame_metric_from_seconds_map(
        anno_root_path, seconds_map_by_thresh,
        downsample=1, rel_dist_list=rel_dist_list,
        filter_low_consis=True, consis_th=0.3, debug=True,
    )
    global_avg = summarize_global_avg_over_thresholds(eval_results, d=0.10)
    print(
        f"[GLOBAL AVG] F1@0.10={global_avg['f1']:.4f} "
        f"(Prec={global_avg['precision']:.4f}, Rec={global_avg['recall']:.4f})"
    )
    return eval_results, float(global_avg['f1'])


@torch.no_grad()
def eval_event_with_external_detector(encoder_model, external_detector, testload, gpu=0):
    old = encoder_model.detector
    encoder_model.detector = external_detector
    try:
        return test_epoch_event(testload, encoder_model, gpu=gpu)
    finally:
        encoder_model.detector = old


@torch.no_grad()
def eval_scene_with_external_detector(encoder_model, external_detector, test_loader, device=DEVICE):
    old = encoder_model.detector
    encoder_model.detector = external_detector
    try:
        return eval_epoch_scene(test_loader, encoder_model, device=device)
    finally:
        encoder_model.detector = old


# =========================
# STAGE 1: Event-Only (from scratch)
# =========================
def run_stage1():
    print("\n" + "=" * 60)
    print("STAGE 1: 事件独立预训练 (产出最佳事件 ckpt 作为下游基础)")
    print("=" * 60)

    args_eval = argparse.Namespace()
    args_eval.feature_path    = kinetics_dataset_path + "features"
    args_eval.annotation_path = kinetics_dataset_path + "data"
    args_eval.score_path      = kinetics_dataset_path + "data"
    args_eval.window_size     = T
    args_eval.interval        = 1
    args_eval.device          = DEVICE

    train_loader_event, event_keep, event_total = build_subset_loader_event(
        EVENT_KEEP_RATIO, BATCH_SIZE, T
    )
    print(f"[Event-pretrain] subset: {event_keep}/{event_total} "
          f"({event_keep/event_total:.1%})")

    val_dataset_event = build_kinetics(split='val', args=args_eval)
    val_loader_event = DataLoader(
        val_dataset_event, batch_size=256, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
        persistent_workers=False,
    )

    student, stu_hook, stu_z_dim = build_student_with_event_head(device=DEVICE, size=T)

    opt_pretrain = torch.optim.Adam(
        student.parameters(),
        lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4
    )
    n_steps0 = len(train_loader_event)
    lr_sh_pretrain = torch.optim.lr_scheduler.LambdaLR(
        opt_pretrain,
        warmup_decay_cosine(n_steps0, n_steps0 * max(EVENT_PRETRAIN_EPOCHS - 1, 1))
    )

    best_event_f1 = None

    for ep in range(EVENT_PRETRAIN_EPOCHS):
        train_epoch_event_only(train_loader_event, student, opt_pretrain,
                               lr_sh_pretrain, device=DEVICE)

        results_dict = test_epoch_event(val_loader_event, student, gpu=GPU)
        eval_res, ev_f1 = evaluate_multi_thresholds(
            results_dict, anno_root_path=args_eval.annotation_path,
            thresholds=[0.10, 0.20, 0.30, 0.40, 0.50],
            rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
            step_frames=1, log_tag=f'stage1_event_ep{ep}'
        )
        print(f"[Stage1 Epoch {ep}] event F1@0.10 = {ev_f1:.4f}")

        save_json(
            {
                "stage": "stage1_event_pretrain",
                "epoch": ep,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "f1_at_0.10_global_avg": ev_f1,
                "event_subset": {"keep": event_keep, "total": event_total,
                                 "ratio": event_keep / event_total},
            },
            os.path.join(CKPT_DIR, f"stage1_event_eval_ep{ep}.json")
        )
        torch.save({"model": student.state_dict(), "epoch": ep},
                   os.path.join(CKPT_DIR, f"stage1_event_ep{ep}.pth"))

        if best_event_f1 is None or ev_f1 > best_event_f1:
            best_event_f1 = ev_f1
            torch.save(
                {"model": student.state_dict(), "epoch": ep, "f1": ev_f1},
                EVENT_ONLY_BEST_PATH
            )
            print(f"[Stage1] new best event F1@0.10={ev_f1:.4f} -> {EVENT_ONLY_BEST_PATH}")

    stu_hook.close()
    print(f"\n[Stage 1 DONE] Best event F1@0.10 = {best_event_f1}")
    print(f"               Saved: {EVENT_ONLY_BEST_PATH}")
    print("Now run:  python train_3stages_event_first.py --stage 2")


# =========================
# STAGE 2: 场景训练 + 事件 teacher KD + 域对抗
# =========================
def run_stage2():
    print("\n" + "=" * 60)
    print("STAGE 2: 场景训练 (以最佳事件 ckpt 为基础) + 事件 KD + 域对抗 (Full SchemeA)")
    print("=" * 60)

    require_ckpt(EVENT_ONLY_BEST_PATH, "2 (Scene + Event teacher)")

    # 主任务: 场景全量
    train_loader_scene_main, scene_keep, scene_total = build_subset_loader_scene(
        SPLIT_PATH_SCENE, SCENE_KEEP_RATIO, BATCH_SIZE, T, mode='train'
    )
    print(f"[Stage2 Scene-main] subset: {scene_keep}/{scene_total} "
          f"({scene_keep/scene_total:.1%})")

    # 辅助: 事件子集 (仅 adv)
    train_loader_event_aux, event_aux_keep, event_aux_total = build_subset_loader_event(
        EVENT_AUX_RATIO, BATCH_SIZE, T
    )
    print(f"[Stage2 Event-aux] subset: {event_aux_keep}/{event_aux_total} "
          f"({event_aux_keep/event_aux_total:.1%})")

    # 测试 loader (注意: 第二次调 load_data, 会复用 load_movienet_server 里的 pkl 缓存)
    test_loader_scene = load_data(
        LABEL_PATH, IMG_PATH, PLC_PATH, SPLIT_PATH_SCENE,
        BATCH_SIZE, seg_sz=T, mode1='test'
    )

    # 学生: encoder ← best event ckpt + MlpHead
    student, stu_hook, stu_z_dim = build_student_with_scene_head_from_event_ckpt(
        event_ckpt_path=EVENT_ONLY_BEST_PATH, device=DEVICE, size=T
    )

    # 教师: encoder ← best event ckpt (frozen)
    teacher, tea_hook, tea_z_dim = build_teacher_encoder_from_event_ckpt(
        ckpt_path=EVENT_ONLY_BEST_PATH, device=DEVICE, size=T
    )
    print("✅ Teacher (event-pretrained encoder) loaded & frozen")

    # 教师事件 detector (cross-task 验证用)
    teacher_event_detector = build_teacher_event_detector_from_ckpt(
        ckpt_path=EVENT_ONLY_BEST_PATH, device=DEVICE
    )
    print("✅ Teacher event detector (FrameWiseHead) loaded & frozen")

    # 判别器
    disc = TemporalDiscriminator(in_dim=stu_z_dim).to(DEVICE)

    opt_main = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad],
        lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4
    )
    opt_disc = torch.optim.Adam(disc.parameters(), lr=1e-4,
                                betas=(0.9, 0.98), weight_decay=1e-4)

    n_steps2 = len(train_loader_scene_main)
    lr_sh_main = torch.optim.lr_scheduler.LambdaLR(
        opt_main, warmup_decay_cosine(n_steps2, n_steps2 * max(SCENE_TRAIN_EPOCHS - 1, 1))
    )
    lr_sh_disc = torch.optim.lr_scheduler.LambdaLR(
        opt_disc, warmup_decay_cosine(n_steps2, n_steps2 * max(SCENE_TRAIN_EPOCHS - 1, 1))
    )

    best_scene_score = None

    # event val loader (用于 ExtraVal1)
    args_eval = argparse.Namespace()
    args_eval.feature_path    = kinetics_dataset_path + "features"
    args_eval.annotation_path = kinetics_dataset_path + "data"
    args_eval.score_path      = kinetics_dataset_path + "data"
    args_eval.window_size     = T
    args_eval.interval        = 1
    args_eval.device          = DEVICE

    for ep in range(SCENE_TRAIN_EPOCHS):
        train_epoch_scene_with_event_teacher(
            train_loader_scene=train_loader_scene_main,
            train_loader_event_aux=train_loader_event_aux,
            student=student, stu_hook=stu_hook, stu_z_dim=stu_z_dim,
            teacher=teacher, tea_hook=tea_hook, tea_z_dim=tea_z_dim,
            disc=disc,
            opt_main=opt_main, opt_disc=opt_disc,
            lr_sh_main=lr_sh_main, lr_sh_disc=lr_sh_disc,
            device=DEVICE,
            alpha_feat=ALPHA_FEAT, lambda_adv=LAMBDA_ADV,
            proj_to_teacher=USE_PROJ_TO_TEACHER,
        )

        # 验证场景
        met, moviePL = eval_epoch_scene(test_loader_scene, student, device=DEVICE)
        print(f"[Stage2 Epoch {ep}] scene metric = {met}")

        save_json(
            {
                "stage": "stage2_scene_with_event_teacher",
                "variant": "full_schemeA",
                "epoch": ep,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metric": met,
                "moviePL": moviePL,
                "scene_subset":     {"keep": scene_keep, "total": scene_total,
                                     "ratio": scene_keep / scene_total},
                "event_aux_subset": {"keep": event_aux_keep, "total": event_aux_total,
                                     "ratio": event_aux_keep / event_aux_total},
                "alpha_feat": ALPHA_FEAT,
                "lambda_adv": LAMBDA_ADV,
                "best_event_ckpt_used_as_basis": EVENT_ONLY_BEST_PATH,
            },
            os.path.join(CKPT_DIR, f"stage2_scene_eval_ep{ep}.json")
        )
        torch.save({"model": student.state_dict(), "epoch": ep},
                   os.path.join(CKPT_DIR, f"stage2_scene_ep{ep}.pth"))

        if isinstance(met, dict) and len(met) > 0:
            key = next(iter(met.keys()))
            score = float(met[key])
            if best_scene_score is None or score > best_scene_score:
                best_scene_score = score
                torch.save(
                    {"model": student.state_dict(), "epoch": ep, "metric": met,
                     "selected_key": key, "selected_score": score},
                    SCENE_WITH_EVENT_TEACHER_BEST_PATH
                )
                print(f"[Stage2] new best scene ({key}={score:.4f}) -> "
                      f"{SCENE_WITH_EVENT_TEACHER_BEST_PATH}")

    # ----- Extra Validation 1: 事件 teacher detector + Stage2 学生 encoder → 评估 event -----
    print("\n" + "=" * 60)
    print("Extra Validation 1: 事件 teacher detector + Stage2 学生 encoder (评估事件)")
    print("=" * 60)

    val_dataset_event = build_kinetics(split='val', args=args_eval)
    val_loader_event = DataLoader(
        val_dataset_event, batch_size=256, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
        persistent_workers=False,
    )

    results_dict_xv1 = eval_event_with_external_detector(
        encoder_model=student,
        external_detector=teacher_event_detector,
        testload=val_loader_event,
        gpu=GPU,
    )
    eval_res_xv1, ev_f1_xv1 = evaluate_multi_thresholds(
        results_dict_xv1, anno_root_path=args_eval.annotation_path,
        thresholds=[0.10, 0.20, 0.30, 0.40, 0.50],
        rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        step_frames=1, log_tag='stage2_extraval_event'
    )
    print(f"[ExtraVal1] event F1@0.10 (teacher head + Stage2 encoder) = {ev_f1_xv1:.4f}")

    save_json(
        {
            "stage": "extra_validation_1_event_detector_plus_stage2_encoder",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "f1_at_0.10_global_avg": ev_f1_xv1,
            "results": eval_res_xv1,
        },
        os.path.join(CKPT_DIR, "extra_validation_1_event_head_stage2.json")
    )

    print(f"\n[Stage 2 DONE] Best scene = {best_scene_score}")
    print(f"               Saved: {SCENE_WITH_EVENT_TEACHER_BEST_PATH}")
    print("Now run:  python train_3stages_event_first.py --stage 3")


# =========================
# STAGE 3: 冻结 backbone + 训新 FrameWiseHead
# =========================
def run_stage3():
    print("\n" + "=" * 60)
    print("STAGE 3: 冻结 backbone, 仅训新的 FrameWiseHead (恢复事件头)")
    print("=" * 60)

    require_ckpt(SCENE_WITH_EVENT_TEACHER_BEST_PATH, "3 (Frozen backbone + new event head)")
    require_ckpt(EVENT_ONLY_BEST_PATH, "3 (need teacher event detector for extra val)")

    args_eval = argparse.Namespace()
    args_eval.feature_path    = kinetics_dataset_path + "features"
    args_eval.annotation_path = kinetics_dataset_path + "data"
    args_eval.score_path      = kinetics_dataset_path + "data"
    args_eval.window_size     = T
    args_eval.interval        = 1
    args_eval.device          = DEVICE

    train_loader_event_full, _, _ = build_subset_loader_event(1.0, BATCH_SIZE, T)
    val_dataset_event = build_kinetics(split='val', args=args_eval)
    val_loader_event = DataLoader(
        val_dataset_event, batch_size=256, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False,
        persistent_workers=False,
    )

    # 学生: 先按 scene-head 结构搭, 再把 Stage2 best 完整加载 (含 MlpHead),
    # 然后冻结全部, 最后替换成全新 FrameWiseHead 并解冻
    student, _stu_hook_old, _ = build_student_with_scene_head_from_event_ckpt(
        event_ckpt_path=EVENT_ONLY_BEST_PATH, device=DEVICE, size=T
    )
    sd = torch.load(SCENE_WITH_EVENT_TEACHER_BEST_PATH, map_location="cpu")
    sd = extract_state_dict(sd)
    sd = strip_module_prefix(sd)
    info = student.load_state_dict(sd, strict=False)
    print("[Stage3] load Stage2 best -> Missing:", info.missing_keys)
    print("[Stage3] load Stage2 best -> Unexpected:", info.unexpected_keys)

    for p in student.parameters():
        p.requires_grad = False
    print("✅ Frozen all parameters")

    student.detector = FrameWiseHead(in_features=2176).to(DEVICE)
    for p in student.detector.parameters():
        p.requires_grad = True
    print("✅ Fresh FrameWiseHead enabled for training")

    # 教师事件 detector (供 ExtraVal2)
    teacher_event_detector = build_teacher_event_detector_from_ckpt(
        ckpt_path=EVENT_ONLY_BEST_PATH, device=DEVICE
    )
    print("✅ Teacher event detector (FrameWiseHead from Stage 1) loaded & frozen")

    opt_event_head = torch.optim.Adam(
        student.detector.parameters(),
        lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4
    )
    n_steps3 = len(train_loader_event_full)
    lr_sh_event_head = torch.optim.lr_scheduler.LambdaLR(
        opt_event_head,
        warmup_decay_cosine(n_steps3, n_steps3 * max(EVENT_HEAD_EPOCHS - 1, 1))
    )

    best_stage3_f1 = None

    for ep in range(EVENT_HEAD_EPOCHS):
        print(f"\n[Stage3 Event-head Epoch {ep}]")
        train_epoch_event_head_only(
            train_loader_event_full, student, opt_event_head,
            lr_sh_event_head, device=DEVICE
        )

        results_dict = test_epoch_event(val_loader_event, student, gpu=GPU)
        eval_res, ev_f1 = evaluate_multi_thresholds(
            results_dict, anno_root_path=args_eval.annotation_path,
            thresholds=[0.10, 0.20, 0.30, 0.40, 0.50],
            rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
            step_frames=1, log_tag=f'stage3_event_ep{ep}'
        )
        print(f"[Stage3 Epoch {ep}] event F1@0.10 = {ev_f1:.4f}")

        save_json(
            {
                "stage": "stage3_event_head_training",
                "epoch": ep,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "f1_at_0.10_global_avg": ev_f1,
            },
            os.path.join(CKPT_DIR, f"stage3_event_eval_ep{ep}.json")
        )
        torch.save({"model": student.state_dict(), "epoch": ep},
                   os.path.join(CKPT_DIR, f"stage3_event_ep{ep}.pth"))

        if best_stage3_f1 is None or ev_f1 > best_stage3_f1:
            best_stage3_f1 = ev_f1
            torch.save(
                {"model": student.state_dict(), "epoch": ep, "f1": ev_f1},
                EVENT_HEAD_BEST_PATH
            )
            print(f"[Stage3] new best event F1@0.10={ev_f1:.4f} -> {EVENT_HEAD_BEST_PATH}")

    # ----- Extra Validation 2: 事件 teacher detector + 最终 encoder → 评估 event -----
    print("\n" + "=" * 60)
    print("Extra Validation 2: 事件 teacher detector + 最终 encoder (评估事件)")
    print("=" * 60)

    results_dict_xv2 = eval_event_with_external_detector(
        encoder_model=student,
        external_detector=teacher_event_detector,
        testload=val_loader_event,
        gpu=GPU,
    )
    eval_res_xv2, ev_f1_xv2 = evaluate_multi_thresholds(
        results_dict_xv2, anno_root_path=args_eval.annotation_path,
        thresholds=[0.10, 0.20, 0.30, 0.40, 0.50],
        rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        step_frames=1, log_tag='stage3_extraval_event'
    )
    print(f"[ExtraVal2] event F1@0.10 (teacher head + final encoder) = {ev_f1_xv2:.4f}")

    save_json(
        {
            "stage": "extra_validation_2_event_detector_plus_final_encoder",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "f1_at_0.10_global_avg": ev_f1_xv2,
            "results": eval_res_xv2,
        },
        os.path.join(CKPT_DIR, "extra_validation_2_event_head_final.json")
    )

    print(f"\n[Stage 3 DONE] Best Stage3 event F1@0.10 = {best_stage3_f1}")
    print(f"               Saved: {EVENT_HEAD_BEST_PATH}")
    print(f"               Checkpoints dir: {CKPT_DIR}")


# =========================
# Entry
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Event-First three-stage independent training (Event -> Scene -> Event)"
    )
    parser.add_argument('--stage', type=int, required=True, choices=[1, 2, 3],
                        help="1: Event-Only / 2: Scene+EventTeacher+Adv / 3: Frozen+New FrameWiseHead")
    args = parser.parse_args()

    if args.stage == 1:
        run_stage1()
    elif args.stage == 2:
        run_stage2()
    elif args.stage == 3:
        run_stage3()


if __name__ == '__main__':
    main()
