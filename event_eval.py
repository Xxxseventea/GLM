"""
evaluate_event_task_full.py
---------------------------
用于加载混合训练后的模型，替换为事件检测头 FrameWiseHead，
并在事件数据集上评估事件检测性能 (F1 / Precision / Recall)。
"""
import torch.multiprocessing as mp
import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from model.CAT_event import EventCAT
from model.backbone.RelationNet_PGM import LocalUncertaintyAwareGraphAttentionLite as LocalUncertaintyAwareGraphAttentionLite_PGM
from model.resnet50_1d import resnet50_1d
from transformers import BertConfig
from model.LGSS import LGSSConfig
from model.backbone.RelationNet_event import LocalUncertaintyAwareGraphAttentionLite
from model.detector.event_detector import FrameWiseHead
from model.lgss_event import LGSSEventDet
from dataset.kinetics.dataset import build, collate_fn
from tool.kinetics_metric import (
    save_submission_in_seconds_multi_thresh,
    eval_by_frame_metric_from_seconds_map,
    summarize_global_avg_over_thresholds,
)

# ============================================================
#   PGM方法 专用
# ============================================================
@torch.no_grad()
def pgm_eval(model, device, val_loader, mask=None):
    """用多阈值 F1 评测 event 检测性能"""
    model.eval()
    results_dict = defaultdict(list)

    for batch in tqdm(val_loader, desc="Eval(Event)"):
        locations, features, targets_list, num_frames, base, coherence = batch
        B, L, _ = features.shape
        features = features.to(device)

        feature, (score_logit, center_pred) = model(
            features, "event", mask=mask, mode='test'
        )
        probs = torch.sigmoid(score_logit)

        base_b = base.view(-1).to(device).float()
        loc = locations.to(device).float()

        t = center_pred.clamp(0, 1) * (L - 1)
        lo = t.floor().long().clamp(0, L - 1)
        hi = (lo + 1).clamp(0, L - 1)
        w = (t - lo.float())
        rel_lo = loc[torch.arange(B, device=device), lo, 0]
        rel_hi = loc[torch.arange(B, device=device), hi, 0]
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

# ------------------------------------------------------
@torch.no_grad()
def test_epoch(testload, model, gpu=0):
    """推理事件数据集并收集边界预测。"""
    model.eval().cuda(gpu)
    results_dict = defaultdict(list)

    for batch in tqdm(testload, desc="Evaluate(Event Dataset)"):
        continue
        locations, features, targets_list, num_frames, base, coherence = batch
        B, L, _ = features.shape
        features = features.cuda(gpu, non_blocking=True)
        # -------------------BECAME方法需用下面的-----------
        score_logit, center_pred = model(features, "event")
#         # -------------------其他------------
#         score_logit, center_pred = model(features)
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
                    "boundaries": abs_frames[i: i + 1].detach().cpu(),
                    "scores": probs[i: i + 1].detach().cpu(),
                }
            )
    return dict(results_dict)


# ------------------------------------------------------
def evaluate_multi_thresholds(
        results,
        anno_root_path,
        thresholds=None,
        rel_dist_list=(0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50),
        step_frames=1,
        log_tag="event_eval",
        name = None,
        mode='eval'

):
    """针对多阈值综合评测事件检测性能。"""
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    seconds_map_by_thresh = save_submission_in_seconds_multi_thresh(
        results,
        anno_root_path,
        thresholds,
        step_frames=step_frames,
        out_dir=f"{log_tag}_pred_outputs",
        write_pkls=False,
        debug=False,
    )

    eval_results = eval_by_frame_metric_from_seconds_map(
        anno_root_path,
        seconds_map_by_thresh,
        downsample=1,
        rel_dist_list=rel_dist_list,
        filter_low_consis=True,
        consis_th=0.3,
        debug=False,
        mode = 'eval',
        name = name

    )

    return eval_results


# ------------------------------------------------------
def main():
    # ===== 命令行参数解析 =====
    parser = argparse.ArgumentParser(description="Event Detection Evaluation Script")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/data/dengyunhui/txx_code/work2/PC/event/ckpt_best.pth",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--feature_path",
        type=str,
        default="/data/shared_dataset/Kinetics/features",
        help="Path to features directory",
    )
    parser.add_argument(
        "--anno_path",
        type=str,
        default="/data/shared_dataset/Kinetics/data",
        help="Path to annotation directory",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU id to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=21,
        help="Window size for dataset",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of heads in model",
    )
    parser.add_argument(
        "--log_tag",
        type=str,
        default="event_eval",
        help="Tag for logging output",
    )

    args = parser.parse_args()
    pth_name = Path(args.ckpt_path).name if 'lambda' in Path(args.ckpt_path).name else Path(args.ckpt_path).stem
    name = pth_name
    print(name)
    # ===== 数据加载 (Event 数据集) =====
    dataset_args = argparse.Namespace()
    dataset_args.feature_path = args.feature_path
    dataset_args.annotation_path = args.anno_path
    dataset_args.score_path = args.anno_path
    dataset_args.window_size = args.window_size
    dataset_args.interval = 1
    dataset_args.device = f"cuda:{args.gpu}"

    print("📦 Building EVENT dataset (validation split)...")
    val_dataset = build(split="val", args=dataset_args)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
# ╔══════════════════════════════════════════════════════════════╗
# ║                               构建模型部分                     ║
# ╚══════════════════════════════════════════════════════════════╝
    # ===== 模型加载 =====
    print(f"\n🚀 Loading backbone weights from {args.ckpt_path} ...")

# --------------------------若要用PC相关的checkpoint，加载此模型---------------------
#     model = resnet50_1d(in_channels=2048, n_classes=1, short_seq=True)
#     ckpt = torch.load(args.ckpt_path, map_location="cpu")
#     model.load_state_dict(ckpt['model'], strict=True)

# --------------------------若要用LGSS相关的checkpoint，加载此模型---------------------
#     model = LGSSEventDet(shot_num=20, place_feat_dim=2048, sim_channel=512)
#     ckpt = torch.load(args.ckpt_path, map_location="cpu")
#     model.load_state_dict(ckpt['model'], strict=True)
#     print("✅ Backbone weights loaded successfully.")
    # --------------------------若要用CAT相关的checkpoint，加载此模型---------------------
#     model_cfg = BertConfig(
#         hidden_size=768, num_hidden_layers=4, num_attention_heads=8,
#         intermediate_size=3072, hidden_act="gelu",
#         hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
#         max_position_embeddings=2048,   # T 上限
#         layer_norm_eps=1e-12,
#     )
#     model_cfg.input_dim = 2048                # 你 shot encoder 输出的 C
#     model_cfg.attention_local_window = 5      # 局部注意力窗口（奇数）
#     model_cfg.num_classes = 1 # 事件类别数
#     model_cfg._attn_implementation = "eager"   # 或 "sdpa" 想用 PyTorch SDPA 的话
#     model = EventCAT(model_cfg)
#     ckpt = torch.load(args.ckpt_path, map_location="cpu")
#     model.load_state_dict(ckpt['model'])
# --------------------------若要用本方法编码器相关的checkpoint，加载此模型---------------------
    #-------------------------------BECAME方法-------------------------------------
#     model = LocalUncertaintyAwareGraphAttentionLite(
#         dim_in=2048,
#         num_heads=args.num_heads,
#         window_size=15,
#         sim_temperature=0.07,
#         neighbor_temp=0.5,
#         uncertainty_mode="variance",
#         use_relative_pos_bias=True,
#         norm="ln",
#         size=args.window_size,
#     ).to(dataset_args.device)
#
#     ckpt = torch.load(args.ckpt_path, map_location="cpu")
#
#     new_ckpt = {}
#     for k, v in ckpt.items():
#         if k.startswith("detector.event."):
#             # 去掉 "detector.event." 前缀，换成 "detector."
#             new_key = k.replace("detector.event.", "detector.")
#             new_ckpt[new_key] = v
#         elif k.startswith("detector.scene."):
#             pass  # 丢弃
#         else:
#             new_ckpt[k] = v
#
#     model.load_state_dict(new_ckpt, strict=True)
    #-------------------------------我们的和EWC方法-------------------------------------
    model = LocalUncertaintyAwareGraphAttentionLite(
        dim_in=2048,
        num_heads=args.num_heads,
        window_size=5,
        sim_temperature=0.07,
        neighbor_temp=0.5,
        uncertainty_mode="variance",
        use_relative_pos_bias=True,
        norm="ln",
        size=args.window_size,
    ).to(dataset_args.device)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt['model']

    new_ckpt = {}
    for k, v in state.items():
        if k.startswith("backbone."):
            # 去掉 "backbone." 前缀
            new_key = k[len("backbone."):]
            new_ckpt[new_key] = v
        elif k.startswith("event_head."):
            # event_head.* -> detector.*
            new_key = k.replace("event_head.", "detector.")
            new_ckpt[new_key] = v
        elif k.startswith("scene_head."):
            pass  # 丢弃
        else:
            new_ckpt[k] = v

    model.load_state_dict(new_ckpt, strict=True)
#-----------------------------------------------------------------------------------------------
#---------------------------PGM方法------------------------------------
#     model = LocalUncertaintyAwareGraphAttentionLite_PGM(
#         dim_in=2048,
#         num_heads=8,
#         window_size=9,
#         sparsity=0.5,
#         size=21,
#     ).to(dataset_args.device)
#
#     ckpt = torch.load(args.ckpt_path, map_location="cpu")
#
#     # 兼容 'model_state' 和 'model' 两种 key
#     if 'model_state' in ckpt:
#         sd = ckpt['model_state']
#     elif 'model' in ckpt:
#         sd = ckpt['model']
#     else:
#         sd = ckpt
#
#     model.load_state_dict(sd, strict=True)
#     print("✅ Backbone weights loaded successfully.")
#
#     # 不管走哪个分支，都保证 per_task_masks 被赋值
#     per_task_masks = ckpt.get('per_task_masks', {}) if isinstance(ckpt, dict) else {}
#     tasks_order = ckpt.get('tasks', None) if isinstance(ckpt, dict) else None
#     # event_best.pth 里存的是 task_name 字段（单任务 ckpt）
#     if tasks_order is None:
#         task_name_in_ckpt = ckpt.get('task_name', None) if isinstance(ckpt, dict) else None
#         if task_name_in_ckpt == 'event':
#             tasks_order = ['event']
#         else:
#             tasks_order = ['event', 'scene']   # 默认顺序
#
#     event_task_id = tasks_order.index('event') if 'event' in tasks_order else 0
#     event_mask = per_task_masks.get(event_task_id, None)

# ╔══════════════════════════════════════════════════════════════╗
# ║                                  检测部分                     ║
# ╚══════════════════════════════════════════════════════════════╝

    #------------------PGM方法推理-----------------------
#     results_dict = pgm_eval(model, device=args.gpu, val_loader=val_loader, mask=event_mask)
    #------------------其他方法推理-----------------------
    print("\n🧩 Running evaluation on Event dataset ...")
    results_dict = test_epoch(val_loader, model, gpu=args.gpu)

    print(f"✅ Collected predictions for {len(results_dict)} videos.")

    # ===== 普通评估性能 =====
    print("\n📊 Running multi-threshold evaluation ...")
    evaluate_multi_thresholds(results_dict, anno_root_path=args.anno_path, log_tag=args.log_tag,name=name)

    print("\n🎯 Evaluation done.")


if __name__ == "__main__":
    mp.set_sharing_strategy('file_system')  # ← 在main()最开头加这一行
    main()
