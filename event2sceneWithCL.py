
import argparse
import os
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
from dataset.movienet.load_movienet_server import load_data
from dataset.kinetics.dataset import build as build_kinetics, collate_fn
from tool.warmup_lr import warmup_decay_cosine
from tool.metric.scene_metric import metric
from tool.metric.kinetics_metric import (
    save_submission_in_seconds_multi_thresh,
    eval_by_frame_metric_from_seconds_map,
    summarize_global_avg_over_thresholds,
)
from teacher_head import build_teacher_event_detector
# =================================================


# =========================
# Config
# =========================
EPOCHS = 5
STAGE_3_EPOCHS = 20
BATCH_SIZE = 128
T = 21
GPU = 0
DEVICE = f"cuda:{GPU}"


movienet_dataset_path = "/mnt/MovieNet/"
kinetics_dataset_path = "/root/autodl-tmp/Kinetics/"
# 场景数据路径（新域）
IMG_PATH = movienet_dataset_path + 'ImageNet_shot.pkl'
PLC_PATH = movienet_dataset_path + 'Places_shot.pkl'
LABEL_PATH = movienet_dataset_path + 'label_endShot.pkl'
SPLIT_PATH_SCENE = movienet_dataset_path + 'split318.json'
MAMBA_PATH = IMG_PATH

# 第一任务（事件）checkpoint：用于加载教师 encoder 权重
ck_path = "/root/autodl-tmp/txx_code/"
CKPT_EVENT = ck_path + "checkpoint_event/ckpt_ep0.pth"

# 输出目录
CKPT_DIR = ck_path + 'mamba/checkpoint_event/WithCL_AdvOnly'
os.makedirs(CKPT_DIR, exist_ok=True)

# 方案A超参
ALPHA_FEAT = 0.5   # 特征蒸馏权重
LAMBDA_ADV = 0.1   # 域对抗权重
SCENE_KEEP_RATIO = 1.0  # 场景抽样比例
EVENT_KEEP_RATIO = 0.30  # 事件抽样比例
USE_PROJ_TO_TEACHER = False  # 若教师/学生 z 维度不一致，设为 True

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


# =========================
# Utils: load backbone-only
# =========================
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


# =========================
# Hook方式获取 encoder 的 pre-head 表征 z
# =========================
class EncoderPreHook:
    """
    使用 detector.forward_pre_hook 捕获传入 detector 的特征 z。
    """
    def __init__(self, model_with_head: nn.Module):
        self.z = None
        def pre_hook(module, inputs):
            z = inputs[0]
            self.z = z.mean(dim=1) if z.dim() == 3 else z
        self.h = model_with_head.detector.register_forward_pre_hook(lambda m, inp: pre_hook(m, inp))

    def close(self):
        self.h.remove()



@torch.no_grad()
def eval_event_with_external_detector(
    encoder_model: nn.Module,
    external_detector: nn.Module,
    testload,
    gpu=0,
):
    """
    用 encoder_model 的 encoder + external_detector 做事件验证。
    会临时替换 detector，验证后恢复原 detector。
    """
    old_detector = encoder_model.detector
    encoder_model.detector = external_detector
    try:
        results = test_epoch_event(testload, encoder_model, gpu=gpu)
    finally:
        encoder_model.detector = old_detector
    return results

# =========================
# Build student/teacher
# =========================
def build_student(
    device: str = DEVICE,
    dim_in: int = 2048,
    num_heads: int = 8,
    window_size: int = 5,
    size: int = 21,
    mlp_hid_dim: int = 512,
    mlp_out_dim: int = 1,
) -> Tuple[nn.Module, EncoderPreHook, int]:
    student = LocalUncertaintyAwareGraphAttentionLite(
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
    student.detector = MlpHead(in_dim=2176, hid_dim=mlp_hid_dim, out_dim=mlp_out_dim).to(device)
    hook = EncoderPreHook(student)
    z_dim = 2176
    return student, hook, z_dim


def build_teacher_encoder(
    ckpt_path: str,
    device: str = DEVICE,
    dim_in: int = 2048,
    num_heads: int = 8,
    window_size: int = 5,
    size: int = 21,
) -> Tuple[nn.Module, EncoderPreHook, int]:
    teacher = LocalUncertaintyAwareGraphAttentionLite(
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
    load_backbone_only(teacher, ckpt_path, detector_prefix="detector.")
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    teacher.detector = nn.Identity().to(device)
    hook = EncoderPreHook(teacher)
    z_dim = 2176
    return teacher, hook, z_dim


# =========================
# GRL (Gradient Reversal Layer)
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
# Data helpers
# =========================
def build_subset_loader_scene(split_json, keep_ratio, batch_size, seg_sz):
    """加载场景数据并按 keep_ratio 采样子集"""
    full_loader = load_data(
        LABEL_PATH, IMG_PATH, PLC_PATH, split_json,
        batch_size, seg_sz=seg_sz, mode1='train'
    )
    dataset = full_loader.dataset
    N = len(dataset)
    keep = max(1, int(N * keep_ratio))
    indices = torch.randperm(N)[:keep].tolist()
    subset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=False, drop_last=True
    )
    return loader, keep, N


def build_subset_loader_event(
    keep_ratio: float,
    batch_size: int,
    seg_sz: int,
    device: str = DEVICE,
):
    """加载事件任务 (Kinetics) 数据并按 keep_ratio 采样子集"""
    args = argparse.Namespace()
    args.feature_path =  kinetics_dataset_path + "features"
    args.score_path =  kinetics_dataset_path + "data"
    args.annotation_path =  kinetics_dataset_path + "data"
    args.window_size = seg_sz
    args.interval = 1
    args.device = device

    dataset_full = build_kinetics(split='train', args=args)
    N = len(dataset_full)
    keep = max(1, int(N * keep_ratio))

    indices = torch.randperm(N)[:keep].tolist()
    subset = torch.utils.data.Subset(dataset_full, indices)
    loader = torch.utils.data.DataLoader(
        subset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
        pin_memory=False, drop_last=True
    )

    print(f"[Event (Kinetics)] subset: {keep}/{N} ({keep/N:.1%})")
    return loader, keep, N


def paired_iter(loader_a, loader_b):
    """交替迭代两个 DataLoader"""
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
# Train / Eval
# =========================
def train_epoch_schemeA_mixed(
    train_loader_scene: DataLoader,
    train_loader_event: DataLoader,
    student: nn.Module, stu_hook: EncoderPreHook, stu_z_dim: int,
    teacher: nn.Module, tea_hook: EncoderPreHook, tea_z_dim: int,
    disc: nn.Module,
    opt_main, opt_disc,
    lr_sh_main=None, lr_sh_disc=None,
    device=DEVICE,
    alpha_feat=ALPHA_FEAT,
    lambda_adv=LAMBDA_ADV,
    proj_to_teacher: bool = USE_PROJ_TO_TEACHER,
):
    """训练混合：场景(新域) 70% + 事件(旧域) 30% with 域对抗"""
    student.train().to(device)
    teacher.eval().to(device)
    disc.train().to(device)

    bce = nn.BCEWithLogitsLoss()

    proj = nn.Identity().to(device)
    if proj_to_teacher or (stu_z_dim != tea_z_dim):
        proj = nn.Linear(stu_z_dim, tea_z_dim, bias=False).to(device)

    it = paired_iter(train_loader_scene, train_loader_event)
    pbar = tqdm(range(len(train_loader_scene)), desc="Train(Scene 70% + Event 30%) + SchemeA")

    for _ in pbar:
        batch_scene, batch_event = next(it)
        (name, img_s, plc_s, label_s, pos, _, _) = batch_scene
        (locations, features, targets_list, num_frames, base, coherence) = batch_event
        features = features.cuda(device, non_blocking=True)

        cls_label, center_gt, has_pos, _ = build_center_reg_labels_from_relative_targets(targets_list)
        cls_label = cls_label.cuda(device, non_blocking=True)

        img_s = img_s.to(device, non_blocking=True)
        plc_s = plc_s.to(device, non_blocking=True)
        label_s = label_s.to(device, non_blocking=True).float()

        w = 0.7
        x_new = w * img_s + (1 - w) * plc_s
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

        # 新任务监督
        logits_new, _ = student(x_new)
        logits_new = logits_new.squeeze(-1)
        loss_task = bce(logits_new, label_s)

        # 特征蒸馏
        with torch.no_grad():
            _, _ = teacher(x_new)
            z_t_new = tea_hook.z
        _, _ = student(x_new)
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

        pbar.set_postfix(task=float(loss_task.item()),
                         feat=float(loss_feat.item()),
                         adv=float(loss_adv.item()),
                         d=float(loss_disc.item()),
                         total=float(loss_total.item()))

def train_epoch_abs_wo_global_disc(
    train_loader_scene: DataLoader,
    train_loader_event: DataLoader,
    student: nn.Module, stu_hook: EncoderPreHook, stu_z_dim: int,
    teacher: nn.Module, tea_hook: EncoderPreHook, tea_z_dim: int,
    disc: nn.Module,
    opt_main, opt_disc,
    lr_sh_main=None, lr_sh_disc=None,
    device=DEVICE,
    alpha_feat=ALPHA_FEAT,
    lambda_adv=LAMBDA_ADV,
    proj_to_teacher: bool = USE_PROJ_TO_TEACHER,
):
    """训练混合：场景(新域) 70% + 事件(旧域) 30% with 域对抗"""
    student.train().to(device)
    teacher.eval().to(device)
    disc.train().to(device)

    bce = nn.BCEWithLogitsLoss()

    proj = nn.Identity().to(device)
    if proj_to_teacher or (stu_z_dim != tea_z_dim):
        proj = nn.Linear(stu_z_dim, tea_z_dim, bias=False).to(device)

    it = paired_iter(train_loader_scene, train_loader_event)
    pbar = tqdm(range(len(train_loader_scene)), desc="Train(Scene 70% + Event 30%) + SchemeA")

    for _ in pbar:
        batch_scene, batch_event = next(it)
        (name, img_s, plc_s, label_s, pos, _, _) = batch_scene
        (locations, features, targets_list, num_frames, base, coherence) = batch_event
        features = features.cuda(device, non_blocking=True)

        cls_label, center_gt, has_pos, _ = build_center_reg_labels_from_relative_targets(targets_list)
        cls_label = cls_label.cuda(device, non_blocking=True)

        img_s = img_s.to(device, non_blocking=True)
        plc_s = plc_s.to(device, non_blocking=True)
        label_s = label_s.to(device, non_blocking=True).float()

        w = 0.7
        x_new = w * img_s + (1 - w) * plc_s
        x_old = features

        
        # with torch.no_grad():
        #     _ = student(x_new)
        #     z_new = stu_hook.z.detach()
        #     _ = student(x_old)
        #     z_old = stu_hook.z.detach()
        # x_dom = torch.cat([z_old, z_new], dim=0)
        # y_dom = torch.cat([
        #     torch.zeros(z_old.size(0), 1, device=device),
        #     torch.ones(z_new.size(0), 1, device=device),
        # ], dim=0)
        # logit_d = disc(x_dom)
        # loss_disc = F.binary_cross_entropy_with_logits(logit_d, y_dom)
        # loss_disc.backward()
        # opt_disc.step()
        if lr_sh_disc is not None:
            lr_sh_disc.step()

        # 2) 更新学生
        opt_main.zero_grad(set_to_none=True)

        # 新任务监督
        logits_new, _ = student(x_new)
        logits_new = logits_new.squeeze(-1)
        loss_task = bce(logits_new, label_s)

        # 特征蒸馏
        with torch.no_grad():
            _, _ = teacher(x_new)
            z_t_new = tea_hook.z
        _, _ = student(x_new)
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

        pbar.set_postfix(task=float(loss_task.item()),
                         feat=float(loss_feat.item()),
                         adv=float(loss_adv.item()),
                        #  d=float(loss_disc.item()),
                         total=float(loss_total.item()))

def train_epoch_abs_wo_disc(
    train_loader_scene: DataLoader,
    train_loader_event: DataLoader,
    student: nn.Module, stu_hook: EncoderPreHook, stu_z_dim: int,
    teacher: nn.Module, tea_hook: EncoderPreHook, tea_z_dim: int,
    disc: nn.Module,
    opt_main, opt_disc,
    lr_sh_main=None, lr_sh_disc=None,
    device=DEVICE,
    alpha_feat=ALPHA_FEAT,
    lambda_adv=LAMBDA_ADV,
    proj_to_teacher: bool = USE_PROJ_TO_TEACHER,
):
    """训练混合：场景(新域) 70% + 事件(旧域) 30% with 域对抗"""
    student.train().to(device)
    teacher.eval().to(device)
    disc.train().to(device)

    bce = nn.BCEWithLogitsLoss()

    proj = nn.Identity().to(device)
    if proj_to_teacher or (stu_z_dim != tea_z_dim):
        proj = nn.Linear(stu_z_dim, tea_z_dim, bias=False).to(device)

    it = paired_iter(train_loader_scene, train_loader_event)
    pbar = tqdm(range(len(train_loader_scene)), desc="Train(Scene 70% + Event 30%) + SchemeA")

    for _ in pbar:
        batch_scene, batch_event = next(it)
        (name, img_s, plc_s, label_s, pos, _, _) = batch_scene
        (locations, features, targets_list, num_frames, base, coherence) = batch_event
        features = features.cuda(device, non_blocking=True)

        cls_label, center_gt, has_pos, _ = build_center_reg_labels_from_relative_targets(targets_list)
        cls_label = cls_label.cuda(device, non_blocking=True)

        img_s = img_s.to(device, non_blocking=True)
        plc_s = plc_s.to(device, non_blocking=True)
        label_s = label_s.to(device, non_blocking=True).float()

        w = 0.7
        x_new = w * img_s + (1 - w) * plc_s
        x_old = features

        
        # with torch.no_grad():
        #     _ = student(x_new)
        #     z_new = stu_hook.z.detach()
        #     _ = student(x_old)
        #     z_old = stu_hook.z.detach()
        # x_dom = torch.cat([z_old, z_new], dim=0)
        # y_dom = torch.cat([
        #     torch.zeros(z_old.size(0), 1, device=device),
        #     torch.ones(z_new.size(0), 1, device=device),
        # ], dim=0)
        # logit_d = disc(x_dom)
        # loss_disc = F.binary_cross_entropy_with_logits(logit_d, y_dom)
        # loss_disc.backward()
        # opt_disc.step()
        if lr_sh_disc is not None:
            lr_sh_disc.step()

        # 2) 更新学生
        opt_main.zero_grad(set_to_none=True)

        # 新任务监督
        logits_new, _ = student(x_new)
        logits_new = logits_new.squeeze(-1)
        loss_task = bce(logits_new, label_s)

        # 特征蒸馏
        with torch.no_grad():
            _, _ = teacher(x_new)
            z_t_new = tea_hook.z
        _, _ = student(x_new)
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

        pbar.set_postfix(task=float(loss_task.item()),
                         feat=float(loss_feat.item()),
                         adv=float(loss_adv.item()),
                        #  d=float(loss_disc.item()),
                         total=float(loss_total.item()))

@torch.no_grad()
def eval_epoch_scene(test_loader, model, device=DEVICE):
    """评估场景检测性能"""
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
    """推理事件数据集并收集边界预测"""
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
                "boundaries": abs_frames[i : i + 1].detach().cpu(),
                "scores": probs[i : i + 1].detach().cpu(),
            })
    return dict(results_dict)


def train_epoch_event(
    trainload,
    model,
    opti,
    lr_sh,
    gpu=0,
    pos_weight=None,
    lambda_reg=1.0,
):
    """训练事件检测 (FrameWiseHead)"""
    model.train().cuda(gpu)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
    l1 = nn.SmoothL1Loss(reduction='none')

    progress = tqdm(trainload, desc="Train(K=1, Center+Score)")

    for sample in progress:
        locations, features, targets_list, num_frames, base, coherence = sample
        features = features.cuda(gpu, non_blocking=True)

        cls_label, center_gt, has_pos, _ = build_center_reg_labels_from_relative_targets(targets_list)
        cls_label = cls_label.cuda(gpu, non_blocking=True)
        center_gt = center_gt.cuda(gpu, non_blocking=True)
        has_pos = has_pos.cuda(gpu, non_blocking=True)

        opti.zero_grad(set_to_none=True)

        score_logit, center_pred = model(features)
        loss_cls = F.binary_cross_entropy_with_logits(score_logit, cls_label.float())

        loss = loss_cls
        loss.backward()
        opti.step()
        if lr_sh is not None:
            lr_sh.step()

        progress.set_postfix(loss=float(loss.item()))


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


# =========================
# Main
# =========================
def main():
    print("\n" + "="*60)
    print("PHASE 1: 场景检测 (Scene) with 域对抗学习")
    print("="*60)

    # 构建场景训练子集 (70%)
    train_loader_scene_70, scene_keep, scene_total = build_subset_loader_scene(
        SPLIT_PATH_SCENE, SCENE_KEEP_RATIO, BATCH_SIZE, T
    )
    print(f"[Scene] subset: {scene_keep}/{scene_total} ({scene_keep/scene_total:.1%})")

    # 构建事件训练子集 (30%)
    train_loader_event_30, event_keep, event_total = build_subset_loader_event(
        EVENT_KEEP_RATIO, BATCH_SIZE, T
    )

    # 场景测试集（全量）
    test_loader = load_data(
        LABEL_PATH, IMG_PATH, PLC_PATH,  SPLIT_PATH_SCENE,
        BATCH_SIZE, seg_sz=T, mode1='test'
    )

    # 构建学生和教师
    student, stu_hook, stu_z_dim = build_student(device=DEVICE, size=T)
    teacher, tea_hook, tea_z_dim = build_teacher_encoder(ckpt_path=CKPT_EVENT, device=DEVICE, size=T)
    teacher_event_detector = build_teacher_event_detector(
    ckpt_path=CKPT_EVENT,
    device=DEVICE,
    in_features=2176,
    save_dir=CKPT_DIR,
)

    # 判别器
    disc = TemporalDiscriminator(in_dim=stu_z_dim).to(DEVICE)

    # 优化器
    opt_main = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad],
        lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4
    )
    opt_disc = torch.optim.Adam(disc.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4)

    lr_sh_main = torch.optim.lr_scheduler.LambdaLR(
        opt_main, warmup_decay_cosine(len(train_loader_scene_70), len(train_loader_scene_70) * max(EPOCHS - 1, 1))
    )
    lr_sh_disc = torch.optim.lr_scheduler.LambdaLR(
        opt_disc, warmup_decay_cosine(len(train_loader_scene_70), len(train_loader_scene_70) * max(EPOCHS - 1, 1))
    )

    best_metric = None
    best_path = os.path.join(CKPT_DIR, "scene_best.pth")

    # 训练场景检测
    for ep in range(EPOCHS):
        train_epoch_schemeA_mixed(
        train_loader_scene=train_loader_scene_70,
        train_loader_event=train_loader_event_30,
        student=student, stu_hook=stu_hook, stu_z_dim=stu_z_dim,
        teacher=teacher, tea_hook=tea_hook, tea_z_dim=tea_z_dim,
        disc=disc,
        opt_main=opt_main, opt_disc=opt_disc,
        lr_sh_main=lr_sh_main, lr_sh_disc=lr_sh_disc,
        device=DEVICE,
        alpha_feat=ALPHA_FEAT,
        lambda_adv=LAMBDA_ADV,
        proj_to_teacher=USE_PROJ_TO_TEACHER
    )

        met, moviePL = eval_epoch_scene(test_loader, student, device=DEVICE)
        print(f"[Epoch {ep}] metric = {met}")
        scene_eval_record = {
            "stage": "phase1_scene_training",
            "epoch": ep,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metric": met,
            "moviePL": moviePL,
            "scene_subset": {
                "keep": scene_keep,
                "total": scene_total,
                "ratio": scene_keep / scene_total,
            },
            "event_subset": {
            "keep": event_keep,
            "total": event_total,
            "ratio": event_keep / event_total,
            },
            "alpha_feat": ALPHA_FEAT,
            "lambda_adv": LAMBDA_ADV,
        }

        save_json(
            scene_eval_record,
            os.path.join(CKPT_DIR, f"scene_eval_ep{ep}.json")
        )

        torch.save({"model": student.state_dict(), "epoch": ep},
               os.path.join(CKPT_DIR, f'scene_ep{ep}.pth'))

        if isinstance(met, dict):
            key = next(iter(met.keys()))
            score = float(met[key])
            if (best_metric is None) or (score > best_metric):
                best_metric = score
                torch.save({"model": student.state_dict(), "epoch": ep}, best_path)
                print(f"New best ({key}={score:.4f}) saved -> {best_path}")

    print("\n" + "="*60)
    print("Extra Validation 1: 教师 detector + Phase 1 学生 encoder")
    print("="*60)

    # 这里先构建事件验证集（如果前面还没建）
    args_eval = argparse.Namespace()
    args_eval.feature_path = kinetics_dataset_path + "features"
    args_eval.annotation_path = kinetics_dataset_path + "data"
    args_eval.score_path = kinetics_dataset_path + "data"
    args_eval.window_size = T
    args_eval.interval = 1
    args_eval.device = DEVICE

    val_dataset_e2s = build_kinetics(split='val', args=args_eval)
    val_loader_e2s = torch.utils.data.DataLoader(
        val_dataset_e2s,
        batch_size=256,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )

    results_teacher_head_phase1 = eval_event_with_external_detector(
        encoder_model=student,
        external_detector=teacher_event_detector,
        testload=val_loader_e2s,
        gpu=GPU,
    )
    print(f"✅ Extra Validation 1 collected predictions for {len(results_teacher_head_phase1)} videos")

    eval_res_teacher_head_phase1 = evaluate_multi_thresholds(
        results_teacher_head_phase1,
        anno_root_path=args_eval.annotation_path,
        thresholds=[0.10, 0.20, 0.30, 0.40, 0.50],
        rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        step_frames=1,
        log_tag='event_eval_teacher_head_phase1',
    )

    extra_val1_record = {
    "stage": "extra_validation_1_teacher_detector_plus_phase1_encoder",
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "num_videos": len(results_teacher_head_phase1),
    "results": eval_res_teacher_head_phase1,
}

    save_json(
        extra_val1_record,
    os.path.join(CKPT_DIR, "extra_validation_1_teacher_head_phase1.json")
    )
    print("\n" + "="*60)
    print("PHASE 2: 事件检测 (Event) - 冻结 Backbone，仅训练 FrameWiseHead")
    print("="*60)

    # 冻结 backbone
    for param in student.parameters():
        param.requires_grad = False
    print("✅ Frozen all student backbone parameters")

    # 替换为 FrameWiseHead
    student.detector = FrameWiseHead(in_features=2176).to(DEVICE)
    for param in student.detector.parameters():
        param.requires_grad = True
    print("✅ FrameWiseHead parameters enabled for training")

    # 重新构建事件 DataLoader
    args = argparse.Namespace()
    args.feature_path =  kinetics_dataset_path + "features"
    args.annotation_path =  kinetics_dataset_path + "data"
    args.score_path = kinetics_dataset_path + "data"
    args.window_size = T
    args.interval = 1
    args.device = DEVICE

    # 构建完整事件训练集和验证集
    # train_dataset_full = build_kinetics(split='train', args=args)
    val_dataset = build_kinetics(split='val', args=args)

    # train_loader_event_full = torch.utils.data.DataLoader(
    #     train_dataset_full, batch_size=BATCH_SIZE, shuffle=True,
    #     collate_fn=collate_fn, num_workers=0, pin_memory=False, drop_last=True
    # )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )

    # 事件检测优化器（仅 detector）
    opt_event = torch.optim.Adam(
        student.detector.parameters(), lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4
    )
    lr_sh_event = torch.optim.lr_scheduler.LambdaLR(
        opt_event, warmup_decay_cosine(len(train_loader_event_30), len(train_loader_event_30) * 19)
    )

    # 训练事件检测
    for ep in range(STAGE_3_EPOCHS):
        print(f"\n[Event Epoch {ep}]")
        train_epoch_event(
            trainload=train_loader_event_30,
            model=student,
            opti=opt_event,
            lr_sh=lr_sh_event,
            gpu=GPU,
            pos_weight=None,
            lambda_reg=1.0
        )

        # 验证
        print(f"[Event Epoch {ep}] Running validation...")
        results_dict = test_epoch_event(val_loader, student, gpu=GPU)
        print(f"✅ Collected predictions for {len(results_dict)} videos")

        # 多阈值评测
        eval_res = evaluate_multi_thresholds(
            results_dict,
            anno_root_path=args.annotation_path,
            thresholds=[0.10, 0.20, 0.30, 0.40, 0.50],
            rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
            step_frames=1,
            log_tag='event_eval',
        )
        event_eval_record = {
    "stage": "phase2_event_training",
    "epoch": ep,
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "num_videos": len(results_dict),
    "results": eval_res,
}

        save_json(
            event_eval_record,
            os.path.join(CKPT_DIR, f"event_eval_ep{ep}.json")
        )

        torch.save({"model": student.state_dict(), "epoch": ep},
                   os.path.join(CKPT_DIR, f'event_ep{ep}.pth'))

        print("\n" + "="*60)

    print("Extra Validation 2: 教师 detector + 当前学生 encoder")
    print("="*60)

    results_teacher_head_final = eval_event_with_external_detector(
        encoder_model=student,
        external_detector=teacher_event_detector,
        testload=val_loader,
        gpu=GPU,
    )
    print(f"✅ Extra Validation 2 collected predictions for {len(results_teacher_head_final)} videos")

    eval_res_teacher_head_final = evaluate_multi_thresholds(
        results_teacher_head_final,
        anno_root_path=args.annotation_path,
        thresholds=[0.10, 0.20, 0.30, 0.40, 0.50],
        rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        step_frames=1,
        log_tag='event_eval_teacher_head_final',
    )
    extra_val2_record = {
    "stage": "extra_validation_2_teacher_detector_plus_final_encoder",
    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "num_videos": len(results_teacher_head_final),
    "results": eval_res_teacher_head_final,
    }

    save_json(
    extra_val2_record,
    os.path.join(CKPT_DIR, "extra_validation_2_teacher_head_final.json")
    )
    print("\n🎯 All training and evaluation done!")


if __name__ == '__main__':
    main()
