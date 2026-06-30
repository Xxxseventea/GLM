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
# =================================================


# =========================
# Config
# =========================
SCENE_PRETRAIN_EPOCHS = 5    # Phase 0: 场景预训练，挑出最佳场景 ckpt
EVENT_TRAIN_EPOCHS    = 5    # Phase 1: 事件训练 (基于最佳场景 ckpt + 场景 KD + adv)
SCENE_HEAD_EPOCHS     = 20   # Phase 2: 冻结 backbone，恢复场景头
BATCH_SIZE = 128
T = 21
GPU = 0
DEVICE = f"cuda:{GPU}"


movienet_dataset_path = "/mnt/MovieNet/"
kinetics_dataset_path = "/root/autodl-tmp/Kinetics/"
IMG_PATH         = movienet_dataset_path + 'ImageNet_shot.pkl'
PLC_PATH         = movienet_dataset_path + 'Places_shot.pkl'
LABEL_PATH       = movienet_dataset_path + 'label_endShot.pkl'
SPLIT_PATH_SCENE = movienet_dataset_path + 'split318.json'
MAMBA_PATH       = IMG_PATH

# 输出目录：scene-first + WithCL_L_G 实验,放到 autodl-fs
CKPT_DIR = '/root/autodl-fs/mamba/checkpoint_scene/SceneFirst_WithCL_L_G'
os.makedirs(CKPT_DIR, exist_ok=True)

# Phase 1 超参
ALPHA_FEAT       = 0.5    # 特征蒸馏权重
LAMBDA_ADV       = 0.1    # 域对抗权重 (通过 GRL)
SCENE_KEEP_RATIO = 1.0    # Phase 0 用：场景预训练抽样比例
EVENT_KEEP_RATIO = 1.0    # Phase 1 主任务：事件抽样比例
SCENE_AUX_RATIO  = 0.30   # Phase 1 辅助：场景仅供域对抗使用
USE_PROJ_TO_TEACHER = False


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


# =========================
# Hook: 截取 detector 之前的 encoder 表征 z
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
# Build encoder/student/teacher (Mamba 完整版)
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


def build_student_with_scene_head(device=DEVICE, size=T, mlp_hid=512, mlp_out=1):
    """Phase 0: encoder + MlpHead, 从头训场景。"""
    student = build_encoder(device=device, size=size)
    student.detector = MlpHead(in_dim=2176, hid_dim=mlp_hid, out_dim=mlp_out).to(device)
    hook = EncoderPreHook(student)
    return student, hook, 2176


def build_student_with_event_head_from_scene_ckpt(scene_ckpt_path, device=DEVICE, size=T):
    """Phase 1: encoder ← 最佳场景 ckpt; detector = FrameWiseHead。"""
    student = build_encoder(device=device, size=size)
    student.detector = FrameWiseHead(in_features=2176).to(device)
    load_backbone_only(student, scene_ckpt_path, detector_prefix="detector.")
    print("✅ Student backbone initialized from best scene ckpt; detector = FrameWiseHead")
    hook = EncoderPreHook(student)
    return student, hook, 2176


def build_teacher_encoder_from_scene_ckpt(ckpt_path, device=DEVICE, size=T):
    """Phase 1: teacher encoder ← 最佳场景 ckpt (frozen)。"""
    teacher = build_encoder(device=device, size=size)
    load_backbone_only(teacher, ckpt_path, detector_prefix="detector.")
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    teacher.detector = nn.Identity().to(device)
    hook = EncoderPreHook(teacher)
    return teacher, hook, 2176


def build_teacher_scene_detector_from_ckpt(ckpt_path, device=DEVICE,
                                           in_dim=2176, hid_dim=512, out_dim=1):
    """从最佳场景 ckpt 提取 MlpHead 作为冻结的场景检测头 (用于 cross-task 验证)。"""
    head = MlpHead(in_dim=in_dim, hid_dim=hid_dim, out_dim=out_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = extract_state_dict(ckpt)
    sd = strip_module_prefix(sd)
    head_sd = {
        k.replace("detector.", "", 1): v
        for k, v in sd.items() if k.startswith("detector.")
    }
    info = head.load_state_dict(head_sd, strict=False)
    print("Teacher scene detector load -> Missing:", info.missing_keys)
    print("Teacher scene detector load -> Unexpected:", info.unexpected_keys)
    print("Loaded scene-head params:", len(head_sd))
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
        subset, batch_size=batch_size, shuffle=True, num_workers=2,
        pin_memory=False, drop_last=True
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
            collate_fn=collate_fn, num_workers=2,
            pin_memory=False, drop_last=True
        )
        print(f"[Event (Kinetics)] full: {N}")
        return loader, N, N

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
# PHASE 0: 场景独立预训练
# =========================
def train_epoch_scene_only(train_loader_scene, student, opti, lr_sh=None, device=DEVICE):
    student.train().to(device)
    bce = nn.BCEWithLogitsLoss()
    pbar = tqdm(train_loader_scene, desc="Phase 0: Train(Scene only)")

    for sample in pbar:
        name, img_s, plc_s, label_s, pos, _, _ = sample
        img_s   = img_s.to(device, non_blocking=True)
        plc_s   = plc_s.to(device, non_blocking=True)
        label_s = label_s.to(device, non_blocking=True).float()
        w = 0.7
        x = w * img_s + (1 - w) * plc_s

        opti.zero_grad(set_to_none=True)
        logits, _ = student(x)
        logits = logits.squeeze(-1)
        loss = bce(logits, label_s)
        loss.backward()
        opti.step()
        if lr_sh is not None:
            lr_sh.step()

        pbar.set_postfix(loss=float(loss.item()))


# =========================
# PHASE 1 训练函数 - 三个变体
#   - 主任务: 事件 BCE        (新域 = 事件)
#   - 辅助:  场景             (旧域,仅供域对抗)
#   - Teacher: 来自最佳场景 ckpt 的 encoder (frozen)
# =========================
def _phase1_event_with_scene_teacher_core(
    train_loader_event: DataLoader,
    train_loader_scene_aux: DataLoader,
    student, stu_hook, stu_z_dim,
    teacher, tea_hook, tea_z_dim,
    disc,
    opt_main, opt_disc,
    lr_sh_main, lr_sh_disc,
    device, alpha_feat, lambda_adv, proj_to_teacher,
    update_global_disc: bool,
    desc: str,
):
    student.train().to(device)
    teacher.eval().to(device)
    disc.train().to(device)

    proj = nn.Identity().to(device)
    if proj_to_teacher or (stu_z_dim != tea_z_dim):
        proj = nn.Linear(stu_z_dim, tea_z_dim, bias=False).to(device)

    it = paired_iter(train_loader_event, train_loader_scene_aux)
    pbar = tqdm(range(len(train_loader_event)), desc=desc)

    for _ in pbar:
        batch_event, batch_scene = next(it)
        (locations, features, targets_list, num_frames, base, coherence) = batch_event
        (name, img_s, plc_s, label_s, pos, _, _) = batch_scene

        features = features.cuda(device, non_blocking=True)
        cls_label, center_gt, has_pos, _ = build_center_reg_labels_from_relative_targets(targets_list)
        cls_label = cls_label.cuda(device, non_blocking=True)

        img_s = img_s.to(device, non_blocking=True)
        plc_s = plc_s.to(device, non_blocking=True)
        w = 0.7
        x_old = w * img_s + (1 - w) * plc_s   # 旧域 = 场景
        x_new = features                       # 新域 = 事件

        # 1) 全局判别器更新 (可选, 对应 abs_wo_global_disc 消融)
        loss_disc_val = None
        if update_global_disc:
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
            loss_disc_val = float(loss_disc.item())
        if lr_sh_disc is not None:
            lr_sh_disc.step()

        # 2) 更新学生
        opt_main.zero_grad(set_to_none=True)

        # 主任务：事件 BCE (新域)
        score_logit, center_pred = student(x_new)
        loss_task = F.binary_cross_entropy_with_logits(score_logit, cls_label.float())

        # 特征蒸馏：在新域(event)输入上,让 student encoder 仍贴近 scene-pretrained teacher
        with torch.no_grad():
            _ = teacher(x_new)
            z_t_new = tea_hook.z
        _ = student(x_new)
        z_s_new = stu_hook.z
        z_s_new_proj = proj(z_s_new)
        loss_feat = F.mse_loss(z_s_new_proj, z_t_new)

        # 域对抗 (GRL)
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

        postfix = dict(
            task=float(loss_task.item()),
            feat=float(loss_feat.item()),
            adv=float(loss_adv.item()),
            total=float(loss_total.item()),
        )
        if loss_disc_val is not None:
            postfix['d'] = loss_disc_val
        pbar.set_postfix(**postfix)


def train_epoch_schemeA_mixed(
    train_loader_event, train_loader_scene_aux,
    student, stu_hook, stu_z_dim,
    teacher, tea_hook, tea_z_dim,
    disc, opt_main, opt_disc,
    lr_sh_main=None, lr_sh_disc=None,
    device=DEVICE, alpha_feat=ALPHA_FEAT, lambda_adv=LAMBDA_ADV,
    proj_to_teacher: bool = USE_PROJ_TO_TEACHER,
):
    """完整方案A：全局 disc 更新 + GRL adv + 场景特征蒸馏。"""
    _phase1_event_with_scene_teacher_core(
        train_loader_event, train_loader_scene_aux,
        student, stu_hook, stu_z_dim,
        teacher, tea_hook, tea_z_dim,
        disc, opt_main, opt_disc,
        lr_sh_main, lr_sh_disc,
        device, alpha_feat, lambda_adv, proj_to_teacher,
        update_global_disc=True,
        desc="Phase 1 [Full]: Event + Scene KD + GlobalDisc + GRL",
    )


def train_epoch_abs_wo_global_disc(
    train_loader_event, train_loader_scene_aux,
    student, stu_hook, stu_z_dim,
    teacher, tea_hook, tea_z_dim,
    disc, opt_main, opt_disc,
    lr_sh_main=None, lr_sh_disc=None,
    device=DEVICE, alpha_feat=ALPHA_FEAT, lambda_adv=LAMBDA_ADV,
    proj_to_teacher: bool = USE_PROJ_TO_TEACHER,
):
    """消融：不更新全局 disc,但保留 GRL 对抗。(默认调用此版,匹配你原 main)"""
    _phase1_event_with_scene_teacher_core(
        train_loader_event, train_loader_scene_aux,
        student, stu_hook, stu_z_dim,
        teacher, tea_hook, tea_z_dim,
        disc, opt_main, opt_disc,
        lr_sh_main, lr_sh_disc,
        device, alpha_feat, lambda_adv, proj_to_teacher,
        update_global_disc=False,
        desc="Phase 1 [WoGlobalDisc]: Event + Scene KD + GRL only",
    )


def train_epoch_abs_wo_disc(
    train_loader_event, train_loader_scene_aux,
    student, stu_hook, stu_z_dim,
    teacher, tea_hook, tea_z_dim,
    disc, opt_main, opt_disc,
    lr_sh_main=None, lr_sh_disc=None,
    device=DEVICE, alpha_feat=ALPHA_FEAT, lambda_adv=LAMBDA_ADV,
    proj_to_teacher: bool = USE_PROJ_TO_TEACHER,
):
    """与 wo_global_disc 同 (你原文件里这两个本来就是同一份, 保留接口一致)。"""
    _phase1_event_with_scene_teacher_core(
        train_loader_event, train_loader_scene_aux,
        student, stu_hook, stu_z_dim,
        teacher, tea_hook, tea_z_dim,
        disc, opt_main, opt_disc,
        lr_sh_main, lr_sh_disc,
        device, alpha_feat, lambda_adv, proj_to_teacher,
        update_global_disc=False,
        desc="Phase 1 [WoDisc]: Event + Scene KD + GRL only",
    )


# =========================
# PHASE 2: 冻结 backbone, 只训新的 MlpHead
# =========================
def train_epoch_scene_head_only(train_loader, model, opti, lr_sh=None, device=DEVICE):
    model.train().to(device)
    bce = nn.BCEWithLogitsLoss()
    progress = tqdm(train_loader, desc="Phase 2: Train(Scene head only)")
    for sample in progress:
        name, img_s, plc_s, label_s, pos, _, _ = sample
        img_s   = img_s.to(device, non_blocking=True)
        plc_s   = plc_s.to(device, non_blocking=True)
        label_s = label_s.to(device, non_blocking=True).float()
        w = 0.7
        x = w * img_s + (1 - w) * plc_s
        opti.zero_grad(set_to_none=True)
        logits, _ = model(x)
        logits = logits.squeeze(-1)
        loss = bce(logits, label_s)
        loss.backward()
        opti.step()
        if lr_sh is not None:
            lr_sh.step()
        progress.set_postfix(loss=float(loss.item()))


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
    """返回 (eval_results, f1@0.10_global_avg)"""
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
def eval_scene_with_external_detector(encoder_model, external_detector, test_loader, device=DEVICE):
    old = encoder_model.detector
    encoder_model.detector = external_detector
    try:
        return eval_epoch_scene(test_loader, encoder_model, device=device)
    finally:
        encoder_model.detector = old


@torch.no_grad()
def eval_event_with_external_detector(encoder_model, external_detector, testload, gpu=0):
    """保留: 临时把 detector 换成 external 做事件评估 (本脚本未使用,留作扩展)。"""
    old = encoder_model.detector
    encoder_model.detector = external_detector
    try:
        return test_epoch_event(testload, encoder_model, gpu=gpu)
    finally:
        encoder_model.detector = old


# =========================
# Main
# =========================
def main():
    # 评估 args (Kinetics)
    args_eval = argparse.Namespace()
    args_eval.feature_path    = kinetics_dataset_path + "features"
    args_eval.annotation_path = kinetics_dataset_path + "data"
    args_eval.score_path      = kinetics_dataset_path + "data"
    args_eval.window_size     = T
    args_eval.interval        = 1
    args_eval.device          = DEVICE

    # ============================================================
    # PHASE 0: 场景独立预训练 → 找最佳场景 ckpt
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 0: 场景独立预训练 (产出最佳场景 ckpt 作为事件训练基础) [Mamba/L_G]")
    print("=" * 60)

    train_loader_scene, scene_keep, scene_total = build_subset_loader_scene(
        SPLIT_PATH_SCENE, SCENE_KEEP_RATIO, BATCH_SIZE, T, mode='train'
    )
    print(f"[Scene-pretrain] subset: {scene_keep}/{scene_total} "
          f"({scene_keep/scene_total:.1%})")

    test_loader_scene = load_data(
        LABEL_PATH, IMG_PATH, PLC_PATH, SPLIT_PATH_SCENE,
        BATCH_SIZE, seg_sz=T, mode1='test'
    )

    student, stu_hook, stu_z_dim = build_student_with_scene_head(device=DEVICE, size=T)

    opt_pretrain = torch.optim.Adam(
        student.parameters(),
        lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4
    )
    n_steps0 = len(train_loader_scene)
    lr_sh_pretrain = torch.optim.lr_scheduler.LambdaLR(
        opt_pretrain,
        warmup_decay_cosine(n_steps0, n_steps0 * max(SCENE_PRETRAIN_EPOCHS - 1, 1))
    )

    best_scene_score = None
    best_scene_path  = os.path.join(CKPT_DIR, "scene_pretrain_best.pth")

    for ep in range(SCENE_PRETRAIN_EPOCHS):
        train_epoch_scene_only(train_loader_scene, student, opt_pretrain,
                               lr_sh_pretrain, device=DEVICE)
        met, moviePL = eval_epoch_scene(test_loader_scene, student, device=DEVICE)
        print(f"[Phase0 Epoch {ep}] scene metric = {met}")

        save_json(
            {
                "stage": "phase0_scene_pretrain",
                "epoch": ep,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metric": met,
                "moviePL": moviePL,
                "scene_subset": {"keep": scene_keep, "total": scene_total,
                                 "ratio": scene_keep / scene_total},
            },
            os.path.join(CKPT_DIR, f"phase0_scene_eval_ep{ep}.json")
        )
        torch.save({"model": student.state_dict(), "epoch": ep},
                   os.path.join(CKPT_DIR, f"phase0_scene_ep{ep}.pth"))

        # 取场景精度最好的 ckpt 作为下游事件训练基础
        if isinstance(met, dict) and len(met) > 0:
            key = next(iter(met.keys()))
            score = float(met[key])
            if (best_scene_score is None) or (score > best_scene_score):
                best_scene_score = score
                torch.save(
                    {"model": student.state_dict(), "epoch": ep, "metric": met,
                     "selected_key": key, "selected_score": score},
                    best_scene_path
                )
                print(f"[Phase0] new best scene ({key}={score:.4f}) -> {best_scene_path}")

    print(f"\n[Phase0] DONE. best scene score = {best_scene_score}")
    print(f"[Phase0] best ckpt -> {best_scene_path}")

    stu_hook.close()
    del student, opt_pretrain, lr_sh_pretrain
    torch.cuda.empty_cache()

    # ============================================================
    # PHASE 1: 事件训练 (基础 = 最佳场景 ckpt) + 场景 KD + 域对抗
    #   默认调用 train_epoch_abs_wo_global_disc (匹配你原 main 的选择)
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 1: 事件训练 (以最佳场景 ckpt 为基础) + 场景 KD + 域对抗 [WoGlobalDisc]")
    print("=" * 60)

    train_loader_event_main, event_keep, event_total = build_subset_loader_event(
        EVENT_KEEP_RATIO, BATCH_SIZE, T
    )
    train_loader_scene_aux, scene_aux_keep, scene_aux_total = build_subset_loader_scene(
        SPLIT_PATH_SCENE, SCENE_AUX_RATIO, BATCH_SIZE, T, mode='train'
    )
    print(f"[Phase1 Scene-aux] subset: {scene_aux_keep}/{scene_aux_total} "
          f"({scene_aux_keep/scene_aux_total:.1%})")

    val_dataset_event = build_kinetics(split='val', args=args_eval)
    val_loader_event = torch.utils.data.DataLoader(
        val_dataset_event, batch_size=256, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=False
    )

    # 学生：encoder ← best scene ckpt + FrameWiseHead
    student, stu_hook, stu_z_dim = build_student_with_event_head_from_scene_ckpt(
        scene_ckpt_path=best_scene_path, device=DEVICE, size=T
    )

    # 教师：encoder ← best scene ckpt (frozen)
    teacher, tea_hook, tea_z_dim = build_teacher_encoder_from_scene_ckpt(
        ckpt_path=best_scene_path, device=DEVICE, size=T
    )
    print("✅ Teacher (scene-pretrained encoder) loaded & frozen")

    # 教师场景 detector (用于 cross-task 验证)
    teacher_scene_detector = build_teacher_scene_detector_from_ckpt(
        ckpt_path=best_scene_path, device=DEVICE,
    )
    print("✅ Teacher scene detector (MlpHead) loaded & frozen")

    disc = TemporalDiscriminator(in_dim=stu_z_dim).to(DEVICE)

    opt_main = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad],
        lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4
    )
    opt_disc = torch.optim.Adam(disc.parameters(), lr=1e-4,
                                betas=(0.9, 0.98), weight_decay=1e-4)

    n_steps1 = len(train_loader_event_main)
    lr_sh_main = torch.optim.lr_scheduler.LambdaLR(
        opt_main, warmup_decay_cosine(n_steps1, n_steps1 * max(EVENT_TRAIN_EPOCHS - 1, 1))
    )
    lr_sh_disc = torch.optim.lr_scheduler.LambdaLR(
        opt_disc, warmup_decay_cosine(n_steps1, n_steps1 * max(EVENT_TRAIN_EPOCHS - 1, 1))
    )

    best_event_f1   = None
    best_event_path = os.path.join(CKPT_DIR, "event_best.pth")

    for ep in range(EVENT_TRAIN_EPOCHS):
        # ↓↓↓ 默认调用 wo_global_disc 消融, 与你原 main 一致;
        #     想用完整版改成 train_epoch_schemeA_mixed 即可
        # train_epoch_schemeA_mixed(
        train_epoch_abs_wo_global_disc(
            train_loader_event=train_loader_event_main,
            train_loader_scene_aux=train_loader_scene_aux,
            student=student, stu_hook=stu_hook, stu_z_dim=stu_z_dim,
            teacher=teacher, tea_hook=tea_hook, tea_z_dim=tea_z_dim,
            disc=disc,
            opt_main=opt_main, opt_disc=opt_disc,
            lr_sh_main=lr_sh_main, lr_sh_disc=lr_sh_disc,
            device=DEVICE,
            alpha_feat=ALPHA_FEAT, lambda_adv=LAMBDA_ADV,
            proj_to_teacher=USE_PROJ_TO_TEACHER,
        )

        # 验证事件
        results_dict = test_epoch_event(val_loader_event, student, gpu=GPU)
        eval_res, ev_f1 = evaluate_multi_thresholds(
            results_dict,
            anno_root_path=args_eval.annotation_path,
            thresholds=[0.10, 0.20, 0.30, 0.40, 0.50],
            rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
            step_frames=1,
            log_tag=f'phase1_event_eval_ep{ep}',
        )

        save_json(
            {
                "stage": "phase1_event_training_scene_teacher",
                "variant": "abs_wo_global_disc",
                "epoch": ep,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_videos": len(results_dict),
                "results": eval_res,
                "f1_at_0.10_global_avg": ev_f1,
                "scene_aux_subset": {"keep": scene_aux_keep, "total": scene_aux_total,
                                     "ratio": scene_aux_keep / scene_aux_total},
                "event_subset":     {"keep": event_keep,     "total": event_total,
                                     "ratio": event_keep / event_total},
                "alpha_feat": ALPHA_FEAT,
                "lambda_adv": LAMBDA_ADV,
                "best_scene_score_used_as_basis": best_scene_score,
                "best_scene_ckpt_used_as_basis":  best_scene_path,
            },
            os.path.join(CKPT_DIR, f"phase1_event_eval_ep{ep}.json")
        )
        torch.save({"model": student.state_dict(), "epoch": ep},
                   os.path.join(CKPT_DIR, f"phase1_event_ep{ep}.pth"))

        if (best_event_f1 is None) or (ev_f1 > best_event_f1):
            best_event_f1 = ev_f1
            torch.save({"model": student.state_dict(), "epoch": ep, "f1": ev_f1},
                       best_event_path)
            print(f"[Phase1] new best event F1@0.10={ev_f1:.4f} -> {best_event_path}")

    # ============================================================
    # Extra Validation 1: 场景 teacher detector + Phase 1 学生 encoder → 评估 scene
    # ============================================================
    print("\n" + "=" * 60)
    print("Extra Validation 1: 场景 teacher detector + Phase 1 学生 encoder (评估场景)")
    print("=" * 60)

    met_xv1, moviePL_xv1 = eval_scene_with_external_detector(
        encoder_model=student,
        external_detector=teacher_scene_detector,
        test_loader=test_loader_scene,
        device=DEVICE,
    )
    print(f"[ExtraVal1] scene metric on Phase1 encoder = {met_xv1}")

    save_json(
        {
            "stage": "extra_validation_1_scene_detector_plus_phase1_encoder",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metric": met_xv1,
            "moviePL": moviePL_xv1,
        },
        os.path.join(CKPT_DIR, "extra_validation_1_scene_head_phase1.json")
    )

    # ============================================================
    # PHASE 2: 冻结 backbone, 重新训场景头 (镜像原 Phase 2)
    # ============================================================
    print("\n" + "=" * 60)
    print("PHASE 2: 冻结 backbone, 仅训新的 MlpHead (恢复场景头)")
    print("=" * 60)

    for p in student.parameters():
        p.requires_grad = False
    print("✅ Frozen all student backbone parameters")

    student.detector = MlpHead(in_dim=2176, hid_dim=512, out_dim=1).to(DEVICE)
    for p in student.detector.parameters():
        p.requires_grad = True
    print("✅ Fresh MlpHead enabled for training")

    train_loader_scene_full, _, _ = build_subset_loader_scene(
        SPLIT_PATH_SCENE, 1.0, BATCH_SIZE, T, mode='train'
    )

    opt_scene_head = torch.optim.Adam(
        student.detector.parameters(),
        lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-4
    )
    n_steps2 = len(train_loader_scene_full)
    lr_sh_scene_head = torch.optim.lr_scheduler.LambdaLR(
        opt_scene_head,
        warmup_decay_cosine(n_steps2, n_steps2 * max(SCENE_HEAD_EPOCHS - 1, 1))
    )

    best_phase2_score = None
    best_phase2_path  = os.path.join(CKPT_DIR, "scene_head_best.pth")

    for ep in range(SCENE_HEAD_EPOCHS):
        print(f"\n[Phase2 Scene-head Epoch {ep}]")
        train_epoch_scene_head_only(
            train_loader_scene_full, student, opt_scene_head, lr_sh_scene_head, device=DEVICE
        )

        met, moviePL = eval_epoch_scene(test_loader_scene, student, device=DEVICE)
        print(f"[Phase2 Epoch {ep}] scene metric = {met}")

        save_json(
            {
                "stage": "phase2_scene_head_training",
                "epoch": ep,
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "metric": met,
                "moviePL": moviePL,
            },
            os.path.join(CKPT_DIR, f"phase2_scene_eval_ep{ep}.json")
        )
        torch.save({"model": student.state_dict(), "epoch": ep},
                   os.path.join(CKPT_DIR, f"phase2_scene_ep{ep}.pth"))

        if isinstance(met, dict) and len(met) > 0:
            key = next(iter(met.keys()))
            score = float(met[key])
            if (best_phase2_score is None) or (score > best_phase2_score):
                best_phase2_score = score
                torch.save({"model": student.state_dict(), "epoch": ep, "metric": met},
                           best_phase2_path)
                print(f"[Phase2] new best scene ({key}={score:.4f}) -> {best_phase2_path}")

    # ============================================================
    # Extra Validation 2: 场景 teacher detector + 最终 encoder → 评估 scene
    # ============================================================
    print("\n" + "=" * 60)
    print("Extra Validation 2: 场景 teacher detector + 最终 encoder (评估场景)")
    print("=" * 60)

    met_xv2, moviePL_xv2 = eval_scene_with_external_detector(
        encoder_model=student,
        external_detector=teacher_scene_detector,
        test_loader=test_loader_scene,
        device=DEVICE,
    )
    print(f"[ExtraVal2] scene metric (teacher head + final encoder) = {met_xv2}")

    save_json(
        {
            "stage": "extra_validation_2_scene_detector_plus_final_encoder",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metric": met_xv2,
            "moviePL": moviePL_xv2,
        },
        os.path.join(CKPT_DIR, "extra_validation_2_scene_head_final.json")
    )

    print("\n🎯 All phases done!")
    print(f"   Phase0 best scene = {best_scene_score}")
    print(f"   Phase1 best event F1@0.10 = {best_event_f1}")
    print(f"   Phase2 best scene = {best_phase2_score}")
    print(f"   ckpt dir: {CKPT_DIR}")


if __name__ == '__main__':
    main()
