import os
import argparse
from collections import defaultdict
import torch
from model.CAT_scene import SceneCAT
from model.resnet50_1d import resnet50_1d
from transformers import BertConfig
from model.LGSS import LGSSConfig
from model.lgss_event import LGSSEventDet
from torch import nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.backbone.RelationNet_PGM import LocalUncertaintyAwareGraphAttentionLite as LocalUncertaintyAwareGraphAttentionLite_PGM
from pathlib import Path
# ===== 模型与工具依赖 =====
from model.backbone.RelationNet import LocalUncertaintyAwareGraphAttentionLite
from model.detector.scene_detector import MlpHead
from dataset.kinetics.dataset import build as build_kinetics, collate_fn
from dataset.movienet.load_movienet_server import load_data
from tool.warmup_lr import warmup_decay_cosine
from tool.scene_metric import metric as scene_metric
#-----------------------PGW版本-----------------------
@torch.no_grad()
def evaluate_PGM(scene_loader, model, gpu=0, name=None, mask=None):
    """Scene Head 推理并计算 metric"""
    model.eval().cuda(gpu)
    predlist, labelist, pathlist = [], [], []
    for batch in tqdm(scene_loader, desc="Scene Eval"):
        names, img_ctx, plc_ctx, label, pos, ids, ind = batch = batch
        img_ctx, plc_ctx = img_ctx.cuda(gpu), plc_ctx.cuda(gpu)
        x = 0.7 * img_ctx + 0.3 * plc_ctx
        _, (logits, center) = model(x, task_name='scene', mask=mask, mode='test')
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        predlist.append(probs)
        labelist.append(label.numpy())
        pathlist.append(names)
    met, moviePL = scene_metric(pathlist, predlist, labelist, name=name)
    return met
@torch.no_grad()
def evaluate_scene(scene_loader, model, gpu=0, name=None):
    """Scene Head 推理并计算 metric"""
    model.eval().cuda(gpu)
    predlist, labelist, pathlist = [], [], []
    for batch in tqdm(scene_loader, desc="Scene Eval"):
        names, img_ctx, plc_ctx, label, pos, ids, ind = batch = batch
        img_ctx, plc_ctx = img_ctx.cuda(gpu), plc_ctx.cuda(gpu)
        x = 0.7 * img_ctx + 0.3 * plc_ctx
        #-------------------PC使用下面-------------
#         _, pred = model(x)
        #-------------------其他使用下面-------------
        _, pred = model(x,task_name='scene')
        predlist.append(pred.cpu().numpy())
        labelist.append(label.numpy())
        pathlist.append(names)
    met, moviePL = scene_metric(pathlist, predlist, labelist, name=name)
    return met


def extract_state_dict(ckpt):
    """
    从 checkpoint 中提取真正的 state_dict。
    兼容:
      - {'model_state_dict': ...}
      - {'state_dict': ...}
      - {'model': ...}
      - 直接是 state_dict
    """
    if isinstance(ckpt, dict):
        if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            return ckpt["model_state_dict"]
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
    return ckpt

def strip_module_prefix(sd: dict):
    """
    去掉 DataParallel/DistributedDataParallel 产生的 'module.' 前缀。
    """
    return {
        (k.replace("module.", "", 1) if k.startswith("module.") else k): v
        for k, v in sd.items()
    }

def load_backbone_only(model: nn.Module, ckpt_path: str, detector_prefix: str = "detector."):
    """
    仅加载除 detector 外的骨干权重。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = extract_state_dict(ckpt)
    # ---- 前后对比 ----
    keys_before = list(sd.keys())
    sd = strip_module_prefix(sd)
    keys_after = list(sd.keys())

    print("=" * 80)
    print(f"strip 前 key 数: {len(keys_before)}, strip 后 key 数: {len(keys_after)}")
    print("=" * 80)
    print(f"{'Before':<60} | After")
    print("-" * 120)
    for kb, ka in zip(keys_before[:20], keys_after[:20]):  # 只看前 20 个
        flag = "  <-- changed" if kb != ka else ""
        print(f"{kb:<60} | {ka}{flag}")
    print("=" * 80)

    pruned = {k: v for k, v in sd.items()}
    load_info = model.load_state_dict(pruned, strict=False)
    print("Missing keys:", load_info.missing_keys)
    print("Unexpected keys:", load_info.unexpected_keys)
    print("Loaded params:", len(pruned))
    return load_info


# ===============================================================
# 主程序
# ===============================================================
def main(args):
    gpu = args.gpu
    movienet_path = args.movienet_path
    ckpt_path = args.ckpt_path
    img_path = args.img_path
    plc_path = args.plc_path
    label_path = args.label_path
    split_path = args.split_path
    window_size = args.window_size
    epochs = args.epochs
# ╔══════════════════════════════════════════════════════════════╗
# ║                               构建模型部分                     ║
# ╚══════════════════════════════════════════════════════════════╝
# --------------------------若要用PC相关的checkpoint，加载此模型---------------------
#     model = resnet50_1d(in_channels=2048, n_classes=1, short_seq=True)
#     ckpt = torch.load(args.ckpt_path, map_location="cpu")
#     model.load_state_dict(ckpt['model'], strict=True)

# --------------------------若要用LGSS相关的checkpoint，加载此模型---------------------
#     model = LGSSEventDet(shot_num=20, place_feat_dim=2048, sim_channel=512)
#     ckpt = torch.load(args.ckpt_path, map_location="cpu")
#     model.load_state_dict(ckpt['model_state_dict'], strict=True)
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
#     model = SceneCAT(model_cfg)
#     ckpt = torch.load(args.ckpt_path, map_location="cpu")
#     model.load_state_dict(ckpt['model_state_dict'])
#  # --------------------------若要用本方法编码器相关的checkpoint，加载此模型---------------------
#     # ==== 加载主干 ====
#     print(f"\n🚀 Loading checkpoint: {ckpt_path}")
#     # -----------------------BECAME-------------------------------
#     model = LocalUncertaintyAwareGraphAttentionLite(
#         dim_in=2048, num_heads=8, window_size=15,
#         sim_temperature=0.07, neighbor_temp=0.5,
#         uncertainty_mode="variance", use_relative_pos_bias=True,
#         norm="ln", size=window_size,
#     )
#     # 权重映射：detector.scene.model -> detector.model
#     ckpt = torch.load(args.ckpt_path, map_location="cpu")
#
#     new_ckpt = {}
#     for k, v in ckpt.items():
#         if k.startswith("detector.scene."):
#             # 去掉 "detector.scene." 前缀，换成 "detector."
#             new_key = k.replace("detector.scene.", "detector.")
#             new_ckpt[new_key] = v
#         elif k.startswith("detector.event."):
#             pass  # 丢弃
#         else:
#             new_ckpt[k] = v
#
#     model.load_state_dict(new_ckpt, strict=True)
    #------------------------- EWC、我们的方法-----------------------
#     model = LocalUncertaintyAwareGraphAttentionLite(
#         dim_in=2048, num_heads=8, window_size=5,
#         sim_temperature=0.07, neighbor_temp=0.5,
#         uncertainty_mode="variance", use_relative_pos_bias=True,
#         norm="ln", size=window_size,
#     )
#     ckpt = torch.load(ckpt_path, map_location='cpu')  # 确保这一行存在
#
#     # 权重映射：detector.scene.model -> detector.model
#     ckpt_state = ckpt
#
#     new_state = {}
#
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     sd = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
#
#     new_state = {}
#     for k, v in sd.items():
#         # 跳过 event_head（当前模型没有这部分）
#         if k.startswith("event_head."):
#             continue
#
#         # backbone.xxx -> xxx
#         if k.startswith("backbone."):
#             k = k[len("backbone."):]
#
#         # scene_head.model.x -> detector.model.x
#         if k.startswith("scene_head."):
#             k = k.replace("scene_head.", "detector.", 1)
#
#         new_state[k] = v
#
#     model.load_state_dict(new_state, strict=True)

    #---------------------------PGM方法------------------------------------
    model = LocalUncertaintyAwareGraphAttentionLite_PGM(
        dim_in=2048,
        num_heads=8,
        window_size=9,
        sparsity=0.5,
        size=21,
    )

#-------------------------------PGM/scene2event/PGM_scene2event_scene_best.pth时，
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt['model'], strict=True)
    # 加载 scene 任务对应的 mask（假设 scene 是 task 0）
    per_task_masks = ckpt.get('per_task_masks', {})
    tasks_order = ckpt.get('tasks', ['scene', 'event'])
    scene_task_id = tasks_order.index('scene') if 'scene' in tasks_order else 0

# 权重映射：detector.scene.model -> detector.model
    pth = Path(ckpt_path)
    name = pth.name
    print(name)
    # 加载检查点


    print("✅ Backbone weights loaded successfully.")
    model.to(f"cuda:{gpu}")
    model.eval()

    # ==== 构建数据 ====
    scene_test_loader = load_data(label_path, img_path, plc_path, split_path,
                                  batch=64, seg_sz=window_size, mode1="test", mode2=None)


# ╔══════════════════════════════════════════════════════════════╗
# ║                                  检测部分                     ║
# ╚══════════════════════════════════════════════════════════════╝

# ---------------------------------PGM用这个----------------------------------
    scene_task_id = tasks_order.index('scene') if 'scene' in tasks_order else 0
    scene_mask = per_task_masks.get(scene_task_id, None)
    scene_metrics = evaluate_PGM(scene_test_loader, model, gpu=gpu, name=name,mask=scene_mask)


# ---------------------------------其他方法用这个----------------------------------
#     scene_metrics = evaluate_scene(scene_test_loader, model, gpu=gpu, name=name)
#


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scene Detection Training and Evaluation')

    # 必需参数
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Checkpoint path (e.g., /root/autodl-tmp/checkpoints/scene/epoch_004.pt)')

    # 可选参数（带默认值）
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index (default: 0)')
    parser.add_argument('--movienet_path', type=str,
                        default="<MovieNet_path>",
                        help='Path to MovieNet dataset')
    parser.add_argument('--img_path', type=str,
                        default="<MovieNet/ImageNet_shot.pkl>",
                        help='Path to image context features')
    parser.add_argument('--plc_path', type=str,
                        default="<MovieNet/Places_shot.pkl>",
                        help='Path to place context features')
    parser.add_argument('--label_path', type=str,
                        default="<MovieNet/label_endShot.pkl>",
                        help='Path to labels')
    parser.add_argument('--split_path', type=str,
                        default="<MovieNet/split318.json>",
                        help='Path to train/test split')
    parser.add_argument('--window_size', type=int, default=21,
                        help='Window size for temporal context (default: 21)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')

    args = parser.parse_args()
    main(args)
