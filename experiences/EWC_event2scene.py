"""
Event (Kinetics) → Scene (MovieNet) 连续学习，EWC 防遗忘
----------------------------------------------------------
照搬参考代码的 ElasticWeightConsolidation 模式：
  • mean 和 fisher 作为 buffer 注册到 model 上（随 state_dict 保存/加载）
  • register_ewc_params(...) 在任务切换时调用一次
  • forward_backward_update(...) 训练时自动叠加 consolidation loss

唯一改动：loss 计算用回调传入，以支持多任务 batch 结构。
"""
import os
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.kinetics.dataset import build as build_kinetics, collate_fn
from dataset.movienet.load_movienet_server import load_data
from model.backbone.RelationNet import LocalUncertaintyAwareGraphAttentionLite
from model.detector.scene_detector import MlpHead
from tool.metric.scene_metric import metric
from tool.metric.kinetics_metric import (
    save_submission_in_seconds_multi_thresh,
    eval_by_frame_metric_from_seconds_map,
    summarize_global_avg_over_thresholds,
)
from tool.warmup_lr import warmup_decay_cosine


# ============================================================
# EWC 包装器（复刻参考实现）
# ============================================================
class ElasticWeightConsolidation:
    """
    与参考代码差异：
      - task_loss 由用户传入的 loss_fn(model, batch) 计算，解耦数据结构
      - Fisher 估计采用 log p(y|x) 的梯度平方，逻辑等价
      - consolidation_loss 用 hasattr 做 per-param 检查，支持新增参数
    """

    def __init__(self, model: nn.Module, lr=1e-4, weight=1e6, device="cuda"):
        self.model = model.to(device)
        self.device = device
        self.weight = weight
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr
        )

    # ---------- 快照当前参数作为旧任务最优点 ----------
    def _update_mean_params(self):
        for name, p in self.model.named_parameters():
            buf = name.replace(".", "__") + "_estimated_mean"
            # 用 register_buffer 会跟 state_dict 一起保存
            self.model.register_buffer(buf, p.data.clone())

    # ---------- 估计 Fisher 对角 ----------
    def _update_fisher_params(self, data_loader, num_batches, loss_fn):
        """
        loss_fn(model, batch) -> 标量 task loss
        Fisher ≈ E[(∂ log p(y|x) / ∂θ)²] ≈ E[(∂ loss / ∂θ)²]
        （BCE / CE loss = -log p，符号不影响平方）
        """
        self.model.train()
        trainable = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        fisher = {n: torch.zeros_like(p) for n, p in trainable}

        n_seen = 0
        for i, batch in enumerate(tqdm(data_loader, desc="Fisher", total=num_batches)):
            if i >= num_batches:
                break
            task_loss = loss_fn(self.model, batch)
            grads = autograd.grad(
                task_loss,
                [p for _, p in trainable],
                retain_graph=False,
                allow_unused=True,
            )
            for (n, _), g in zip(trainable, grads):
                if g is not None:
                    fisher[n] += g.detach() ** 2
            n_seen += 1

        for n in fisher:
            fisher[n] /= max(1, n_seen)

        for name, _ in trainable:
            buf = name.replace(".", "__") + "_estimated_fisher"
            self.model.register_buffer(buf, fisher[name])

        print(f"✅ Fisher computed on {n_seen} mini-batches.")

    # ---------- 对外接口 ----------
    def register_ewc_params(self, data_loader, num_batches, loss_fn):
        """任务切换前调用：先算 Fisher（用旧参数），再快照 mean。"""
        self._update_fisher_params(data_loader, num_batches, loss_fn)
        self._update_mean_params()

    def consolidation_loss(self):
        """EWC 二次惩罚项 (λ/2) Σ F_i (θ_i − θ*_i)²"""
        loss = 0.0
        for name, p in self.model.named_parameters():
            base = name.replace(".", "__")
            mean_buf, fisher_buf = base + "_estimated_mean", base + "_estimated_fisher"
            if hasattr(self.model, mean_buf) and hasattr(self.model, fisher_buf):
                mean = getattr(self.model, mean_buf)
                fisher = getattr(self.model, fisher_buf)
                loss = loss + (fisher * (p - mean) ** 2).sum()
        if isinstance(loss, float):
            return torch.tensor(0.0, device=self.device)
        return (self.weight / 2.0) * loss

    def forward_backward_update(self, batch, loss_fn):
        """一步训练：task_loss + EWC → 反传 → 更新"""
        task_loss = loss_fn(self.model, batch)
        reg_loss = self.consolidation_loss()
        total = task_loss + reg_loss

        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()
        return float(total.item()), float(task_loss.item()), float(reg_loss.item())

    def rebuild_optimizer(self, lr=1e-4):
        """切换任务、替换 head 之后重建 optimizer，接管新的可训练参数。"""
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr
        )

    def save(self, filename):
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        self.model.load_state_dict(torch.load(filename, map_location=self.device))

# ============================================================
# 统一模型：一个 forward 同时服务 Event 和 Scene
# ============================================================
class UnifiedBoundaryModel(nn.Module):
    """
    encoder 输出 (score, center)；同时保留一个 scene_head 用于 Scene 阶段。
    两个任务共用 encoder 的参数 —— 这样 EWC 才有意义。
    """

    def __init__(self, encoder: nn.Module, scene_hidden=2176, scene_mid=512):
        super().__init__()
        self.encoder = encoder
        # Scene 头单独一路，Event 阶段 fisher/mean 会包含它的参数
        # 但因为 Event loss 不经过它，对应 fisher=0，不会产生约束
        self.scene_head = MlpHead(scene_hidden, scene_mid, 1)

    def forward_event(self, x):
        """Event 阶段用：返回 encoder 自带的 (score_logit, center_pred)"""
        return self.encoder(x)

    def forward_scene(self, x):
        """
        Scene 阶段用：encoder 抽特征后过 scene_head
        注意: 这里假设 encoder 能给出可供 scene_head 消费的中间表示。
        如果 encoder 内部没有暴露中间特征接口，需要在 encoder 里加一个
        `return_features=True` 分支，或用 hook 拿 penultimate feature。
        """
        # TODO(你来确认): 这里要换成真正的中间特征，不能直接用 score_logit
        score, center = self.encoder(x)
        return score, center  # 如果 encoder 已经能直出 Scene 需要的输出则保留

    # 默认 forward 走 event，给 EWC Fisher 估计用
    def forward(self, x):
        return self.forward_event(x)


# ============================================================
# Batch → loss 的回调函数
# ============================================================
def build_event_cls_label(targets_list, device):
    B = len(targets_list)
    cls = torch.zeros(B, device=device)
    for i, t in enumerate(targets_list):
        if t["boundaries"].numel() > 0:
            cls[i] = 1.0
    return cls


def event_loss_fn(model, batch):
    """Event 任务 loss：窗口内是否存在边界的 BCE"""
    _, features, targets_list, *_ = batch
    device = next(model.parameters()).device
    features = features.to(device)
    cls_label = build_event_cls_label(targets_list, device)
    score_logit, _ = model.forward_event(features)
    return F.binary_cross_entropy_with_logits(score_logit, cls_label)


def scene_loss_fn(model, batch):
    """Scene 任务 loss：MovieNet shot-level 二分类"""
    _, img_ctx, plc_ctx, label, _, _, _ = batch
    device = next(model.parameters()).device
    img_ctx, plc_ctx = img_ctx.to(device), plc_ctx.to(device)
    label = label.to(device).float()
    x = 0.7 * img_ctx + 0.3 * plc_ctx
    logits, _ = model.forward_scene(x)
    return F.binary_cross_entropy_with_logits(logits.squeeze(), label.squeeze())


# ============================================================
# 评测（与前版相同，略作整理）
# ============================================================
@torch.no_grad()
def eval_scene(loader, model, device):
    model.eval()
    preds, labels, paths, ids = [], [], [], []
    for sample in tqdm(loader, desc="Scene Eval"):
        names, img_ctx, plc_ctx, label, _, _, ind = sample
        img_ctx, plc_ctx = img_ctx.to(device), plc_ctx.to(device)
        x = 0.7 * img_ctx + 0.3 * plc_ctx
        _, pred = model.forward_scene(x)
        preds.append(pred.squeeze().cpu().numpy())
        labels.append(label.numpy())
        paths.append(names)
        ids.append(np.asarray(ind))
    return metric(paths, preds, labels)


@torch.no_grad()
def eval_event(loader, model, device):
    model.eval()
    results = defaultdict(list)
    for batch in tqdm(loader, desc="Event Eval"):
        locations, features, targets_list, num_frames, base, coherence = batch
        B, L, _ = features.shape
        features = features.to(device, non_blocking=True)
        score_logit, center_pred = model.forward_event(features)
        probs = torch.sigmoid(score_logit)

        base_b = base.view(-1).to(device).float()
        loc = locations.to(device).float()
        t = center_pred.clamp(0, 1) * (L - 1)
        lo = t.floor().long().clamp(0, L - 1)
        hi = (lo + 1).clamp(0, L - 1)
        w = t - lo.float()
        rel_lo = loc[torch.arange(B), lo, 0]
        rel_hi = loc[torch.arange(B), hi, 0]
        abs_frames = base_b + rel_lo * (1 - w) + rel_hi * w

        for i, td in enumerate(targets_list):
            vt = td.get("video_id", None)
            vid = int(vt.item()) if torch.is_tensor(vt) else (vt if vt is not None else i)
            results[str(vid)].append({
                "boundaries": abs_frames[i:i + 1].detach().cpu(),
                "scores": probs[i:i + 1].detach().cpu(),
            })
    return dict(results)


# def evaluate_multi_thresholds(results, anno_root,
#                               thresholds=(0.10, 0.20, 0.30, 0.40, 0.50),
#                               rel_d=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)):
#     sec_map = save_submission_in_seconds_multi_thresh(
#         results, anno_root, list(thresholds), step_frames=1,
#         out_dir="event_pred_outputs", write_pkls=False, debug=False,
#     )
#     ev = eval_by_frame_metric_from_seconds_map(
#         anno_root, sec_map, downsample=1,
#         rel_dist_list=list(rel_d),
#         filter_low_consis=True, consis_th=0.3, debug=False,
#     )
#     g = summarize_global_avg_over_thresholds(ev, d=0.10)
#     print(f"[Event] F1@0.10={g['f1']:.4f}  P={g['precision']:.4f}  R={g['recall']:.4f}")
#     return ev

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
# 配置
# ============================================================
class CFG:
    DEVICE = "cuda:0"
    T = 21
    EVENT_EPOCHS = 5
    SCENE_EPOCHS = 1
    EVENT_BS = 256
    SCENE_BS = 128
    FISHER_BATCHES = 300        # 参考代码也是 300
    LAMBDA = 1e6                 # 参考代码默认 1_000_000
    LR = 1e-4

    movienet_dataset_path = "/data/shared_dataset/shared_dataset/MovieDatasets/MovieNet/"
    kinetics_dataset_path = "/data/shared_dataset/Kinetics/"
    FEATURE_PATH = kinetics_dataset_path + "features"
    SCORE_PATH =  kinetics_dataset_path + "data"
    ANNOTATION_PATH = kinetics_dataset_path +  "data"
    IMG_PATH = movienet_dataset_path + "ImageNet_shot.pkl"
    PLC_PATH = movienet_dataset_path + "Places_shot.pkl"
    LABEL_PATH = "/home/tianxiaoxuan/data/mamba/data/label_endShot.pkl"

    SPLIT_PATH = "/home/tianxiaoxuan/data/mamba/data/split318.json"

    OUT = "/home/tianxiaoxuan/data/mamba/checkpoint_scene2event_ewc"


# ============================================================
# 主流程
# ============================================================
def main():
    os.makedirs(CFG.OUT, exist_ok=True)

    # ---- 数据 ----
    args = argparse.Namespace(
        feature_path=CFG.FEATURE_PATH, score_path=CFG.SCORE_PATH,
        annotation_path=CFG.ANNOTATION_PATH, window_size=CFG.T,
        interval=1, device=CFG.DEVICE,
    )
    event_train_ds = build_kinetics(split="train", args=args)
    event_val_ds = build_kinetics(split="val", args=args)

    event_train_loader = DataLoader(
        event_train_ds, batch_size=CFG.EVENT_BS, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    event_val_loader = DataLoader(
        event_val_ds, batch_size=256, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    scene_train_loader = load_data(
        CFG.LABEL_PATH, CFG.IMG_PATH, CFG.PLC_PATH, CFG.SPLIT_PATH,
        CFG.SCENE_BS, seg_sz=CFG.T, mode1="train", mode2=None,
    )
    scene_test_loader = load_data(
        CFG.LABEL_PATH, CFG.IMG_PATH, CFG.PLC_PATH, CFG.SPLIT_PATH,
        CFG.SCENE_BS, seg_sz=CFG.T, mode1="test", mode2=None,
    )

    # ---- 模型 + EWC 包装 ----
    encoder = LocalUncertaintyAwareGraphAttentionLite(
        dim_in=2048, num_heads=8, window_size=5,
        sim_temperature=0.07, neighbor_temp=0.5,
        uncertainty_mode="variance", use_relative_pos_bias=True,
        norm="ln", size=CFG.T,
    )
    model = UnifiedBoundaryModel(encoder)
    ewc = ElasticWeightConsolidation(model, lr=CFG.LR, weight=CFG.LAMBDA, device=CFG.DEVICE)

    # ========== 任务 A: Event ==========
    print("\n========== Task A: Event ==========")
    for ep in range(CFG.EVENT_EPOCHS):
        for batch in tqdm(event_train_loader, desc=f"[Event ep{ep}]"):
            ewc.forward_backward_update(batch, event_loss_fn)
        ewc.save(os.path.join(CFG.OUT, f"event_ep{ep}.pt"))

    # ---- 关键:训练完 A 之后注册 Fisher + mean ----
    print("\n>>> Registering EWC params on Scene task ...")
    ewc.register_ewc_params(scene_train_loader, CFG.FISHER_BATCHES, scene_loss_fn)
    ewc.save(os.path.join(CFG.OUT, "after_scene_with_ewc.pt"))

    # ========== 任务 B: Scene ==========
    # 切任务前重建 optimizer（尤其当你想冻结/解冻某些参数时）
    ewc.rebuild_optimizer(lr=CFG.LR)

    print("\n========== Task B: Scene (with EWC penalty) ==========")
    for ep in range(CFG.SCENE_EPOCHS):
        pbar = tqdm(scene_train_loader, desc=f"[Scene ep{ep}]")
        for batch in pbar:
            total, task, reg = ewc.forward_backward_update(batch, scene_loss_fn)
            pbar.set_postfix(total=total, task=task, ewc=reg)

        ewc.save(os.path.join(CFG.OUT, f"scene_ep{ep}.pt"))
        met, _ = eval_scene(scene_test_loader, ewc.model, CFG.DEVICE)
        print(f"[Scene ep{ep}] {met}")

    # ---- EVent 训练完后，回头评测 Scene看遗忘程度 ----
    print("\n>>> Evaluating Scene performance after Scene training (forgetting check) ...")
    met, _ = eval_scene(scene_test_loader, ewc.model, CFG.DEVICE)
    print(f"[Scene ep] {met}")

    # 如果之后还有任务 C，就再 register_ewc_params 一次，新的 Fisher 会覆盖旧的
    # ewc.register_ewc_params(scene_train_loader, CFG.FISHER_BATCHES, scene_loss_fn)


if __name__ == "__main__":
    main()
