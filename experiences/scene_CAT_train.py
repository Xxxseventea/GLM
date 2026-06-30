import os
import glob

from torch.nn import Linear
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset.movienet.load_movienet_server import load_data
from tool.warmup_lr import warmup_decay_cosine
from tool.metric.scene_metric import metric
from datetime import datetime

import json
from model.backbone.RelationNet import LocalUncertaintyAwareGraphAttentionLite
from model.detector.scene_detector import MlpHead
from model.CAT_scene import  SceneCAT
from transformers import BertConfig
# 配置
EPOCHS = 20
BATCH_SIZE = 128
T = 21
GPU = 0

# 路径
movienet_path = "/mnt/MovieNet/"
IMG_PATH = movienet_path + 'ImageNet_shot.pkl'
PLC_PATH = movienet_path + 'Places_shot.pkl'
LABEL_PATH = movienet_path + 'label_endShot.pkl'
SPLIT_PATH = movienet_path + 'split318.json'
MAMBA_PATH = movienet_path + 'ImageNet_shot.pkl'

CKPT_DIR = '/root/autodl-fs/CAT/scene/'
SIDE_DIR = CKPT_DIR + 'side_outputs'
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SIDE_DIR, exist_ok=True)
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

def train_epoch(trainload, model, opti, lr_sh, gpu=GPU):
    model.train()
    progress = tqdm(trainload)
    for i, sample in enumerate(progress):
        names, img_ctx, plc_ctx, label, pos, ids, ind = sample
        opti.zero_grad()

        img_ctx = img_ctx.cuda(gpu)
        plc_ctx = plc_ctx.cuda(gpu)

        w = 0.7  # 例如更信任 x1
        x_out = w * img_ctx + (1 - w) * plc_ctx
        label = label.cuda(gpu)

        # # 前向：返回主输出与旁路 aux；训练忽略 aux
        # # pred = model(img_ctx, plc_ctx, mamba_ctx)
        logtis, pred = model(x_out)
        pred = pred.squeeze()
        logtis = logtis.squeeze()
        # 单一 BCE
        loss= F.binary_cross_entropy_with_logits(logtis, label.float())

        loss.backward()
        opti.step()
        lr_sh.step()
        progress.set_postfix(loss=f'{loss.item():.6f}')
    return 1


def test_epoch(testload, model, gpu=GPU, side_save_path=None):
    predlist, labelist, pathlist, idlist = [], [], [],[]
    side_rel_feat = []

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(testload)):
            names, img_ctx, plc_ctx, label, pos, ids, ind = sample
            img_ctx = img_ctx.cuda(gpu)
            labelist.append(label.numpy())  # 先缓存，后统一拼接
            plc_ctx = plc_ctx.cuda(gpu)

            w = 0.7  # 例如更信任 x1
            x_out = w * img_ctx + (1 - w) * plc_ctx
            # pred= model(img_ctx, plc_ctx, mamba_ctx)
            logtis,pred = model(x_out)
            pred = pred.squeeze()

            predlist.append(pred.cpu().numpy())
            pathlist.append(names)
            idlist.append(np.asarray(ind))  # sample_idx 是 [B] 的张量/列表



    met, moviePL = metric(pathlist, predlist, labelist)

    if side_save_path is not None:
        labels_arr = np.concatenate(labelist)
        probs_arr = np.concatenate(predlist)
        ids_arr = np.concatenate(idlist)  # -> ndarray
        rel_feat_arr = np.concatenate(side_rel_feat) if len(side_rel_feat) else np.array([])
        paths = [p for batch_names in pathlist for p in batch_names]
        np.savez(side_save_path,
                 labels=labels_arr.astype(np.float32),
                 probs=probs_arr.astype(np.float32),
                 rel_feat=rel_feat_arr.astype(np.float32),
                 paths=np.array(paths),
                 ids=ids_arr.astype(np.int64))
    return met, moviePL

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


def main():
    # model = CombinedModel().to(f'cuda:{GPU}')

    model_cfg = BertConfig(
        hidden_size=768, num_hidden_layers=4, num_attention_heads=8,
        intermediate_size=3072, hidden_act="gelu",
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        max_position_embeddings=2048,   # T 上限
        layer_norm_eps=1e-12,
    )
    model_cfg.input_dim = 2048                # 你 shot encoder 输出的 C
    model_cfg.attention_local_window = 5      # 局部注意力窗口（奇数）
    model_cfg.num_classes = 1 # 事件类别数
    model_cfg._attn_implementation = "eager"   # 或 "sdpa" 想用 PyTorch SDPA 的话

    model = SceneCAT(model_cfg)
    model = model.to(f"cuda:{GPU}")

    train_dataLoader = load_data(LABEL_PATH, IMG_PATH, PLC_PATH, SPLIT_PATH,
                         BATCH_SIZE, seg_sz=T, mode1='train', mode2=None)
    test_dataLoader = load_data(LABEL_PATH, IMG_PATH, PLC_PATH, SPLIT_PATH,
                        BATCH_SIZE, seg_sz=T, mode1='test', mode2=None)

    # default
    optimizer = torch.optim.Adam(
        model.parameters(),          # 所有参数一视同仁，wd=0 不分组
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )

    steps_per_epoch = len(train_dataLoader)
    total_steps     = steps_per_epoch * 20        # max_epochs=20

    # warmup=0 → 直接 cosine
    lr_sh = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=0.0)
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=1e-4,
    #     betas=(0.9, 0.98),
    #     weight_decay=1e-4,
    # )
    # lr_sh = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer,
    #     warmup_decay_cosine(len(train_dataLoader), len(train_dataLoader) * (EPOCHS - 1))
    # )

    for i in range(EPOCHS):
        train_epoch(trainload=train_dataLoader, model=model, opti=optimizer, lr_sh=lr_sh, gpu=GPU)
        ckpt_path = os.path.join(CKPT_DIR, f'epoch_{i:03d}.pt')

        ckpt = {
            "epoch": i,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_sh.state_dict() if lr_sh is not None else None,
            "config": {
                "EPOCHS": EPOCHS,
                "BATCH_SIZE": BATCH_SIZE,
                "T": T,
                "GPU": GPU,
                "model_hparams": {
                    "dim_in": 2048,
                    "place_dim": 2048,
                    "num_heads": 8,
                    "window_size": 5,
                    "sim_temperature": 0.07,
                    "neighbor_temp": 0.5,
                    "uncertainty_mode": "variance",
                    "use_relative_pos_bias": True,
                    "norm": "ln",
                },
            },
        }
        torch.save(ckpt, ckpt_path)

        side_path = os.path.join(SIDE_DIR, f'epoch_{i:03d}.npz')
        met, moviePL = test_epoch(testload=test_dataLoader, model=model, gpu=GPU, side_save_path=side_path)
        scene_eval_record = {
            "stage": "phase1_scene_training",
            "epoch": i,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metric": met,
            "moviePL": moviePL,
        }

        save_json(
            scene_eval_record,
            os.path.join(CKPT_DIR, f"scene_eval_ep{i}.json")
        )
        print(f"[Epoch {i}] metric = {met}")


if __name__ == '__main__':
    main()

