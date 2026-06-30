# import os
# import glob
# from tqdm import tqdm
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# from tool.warmup_lr import warmup_decay_cosine
# from tool.metric import metric
# from tool.loss import bce
# from dataset.load_movienet_server import load_data
# from model.RelationNet import LocalUncertaintyAwareGraphAttentionLite
#
#
# # 配置
# EPOCHS = 5
# BATCH_SIZE = 128
# T = 21
# GPU = 6
#
# # 路径
# IMG_PATH = '/data/shared_dataset/shared_dataset/MovieDatasets/MovieNet/ImageNet_shot.pkl'
# PLC_PATH = '/data/shared_dataset/shared_dataset/MovieDatasets/MovieNet/Places_shot.pkl'
# LABEL_PATH = '/home/tianxiaoxuan/data/mamba/data/label_endShot.pkl'
# SPLIT_PATH = '/home/tianxiaoxuan/data/mamba/data/split318.json'
# MAMBA_PATH = '/home/tianxiaoxuan/data/mamba/data/ImageNet_shot.pkl'
#
# CKPT_DIR = '/home/tianxiaoxuan/data/mamba/checkpoint_scene'
# SIDE_DIR = '/home/tianxiaoxuan/data/mamba/side_outputs'
# os.makedirs(CKPT_DIR, exist_ok=True)
# os.makedirs(SIDE_DIR, exist_ok=True)
# import torch
# import numpy as np
# from tqdm import tqdm
# import torch.nn.functional as F
#
# def train_epoch(trainload, model, opti, lr_sh, gpu=GPU):
#     model.train()
#     progress = tqdm(trainload)
#     for i, sample in enumerate(progress):
#         name, img_ctx, plc_ctx, mamba_ctx, label, pos, _,_ = sample
#         opti.zero_grad()
#
#         img_ctx = img_ctx.cuda(gpu)
#         plc_ctx = plc_ctx.cuda(gpu)
#         mamba_ctx = mamba_ctx.cuda(gpu)  # 未使用，但保持接口
#         label = label.cuda(gpu)
#
#         # # 前向：返回主输出与旁路 aux；训练忽略 aux
#         # # pred = model(img_ctx, plc_ctx, mamba_ctx)
#         pred = model(img_ctx,plc_ctx)
#         pred = pred.squeeze()
#         # 单一 BCE
#         loss= F.binary_cross_entropy(pred, label.float())
#
#         loss.backward()
#         opti.step()
#         lr_sh.step()
#         progress.set_postfix(loss=f'{loss.item():.6f}')
#     return 1
#
#
# def test_epoch(testload, model, gpu=GPU, side_save_path=None):
#     predlist, labelist, pathlist, idlist = [], [], [],[]
#     side_rel_feat = []
#
#     model.eval()
#     with torch.no_grad():
#         for i, sample in enumerate(tqdm(testload)):
#             names, img_ctx, plc_ctx, mamba_ctx, label, pos, ids,ind = sample
#             img_ctx = img_ctx.cuda(gpu)
#             plc_ctx = plc_ctx.cuda(gpu)
#             mamba_ctx = mamba_ctx.cuda(gpu)
#             labelist.append(label.numpy())  # 先缓存，后统一拼接
#
#             # pred= model(img_ctx, plc_ctx, mamba_ctx)
#             pred = model(img_ctx,plc_ctx)
#             pred = pred.squeeze()
#             predlist.append(pred.cpu().numpy())
#             pathlist.append(names)
#             idlist.append(np.asarray(ind))  # sample_idx 是 [B] 的张量/列表
#
#
#
#     met, moviePL = metric(pathlist, predlist, labelist)
#
#     if side_save_path is not None:
#         labels_arr = np.concatenate(labelist)
#         probs_arr = np.concatenate(predlist)
#         ids_arr = np.concatenate(idlist)  # -> ndarray
#         rel_feat_arr = np.concatenate(side_rel_feat) if len(side_rel_feat) else np.array([])
#         paths = [p for batch_names in pathlist for p in batch_names]
#         np.savez(side_save_path,
#                  labels=labels_arr.astype(np.float32),
#                  probs=probs_arr.astype(np.float32),
#                  rel_feat=rel_feat_arr.astype(np.float32),
#                  paths=np.array(paths),
#                  ids=ids_arr.astype(np.int64))
#     return met, moviePL
# def main():
#     # model = CombinedModel().to(f'cuda:{GPU}')
#
#
#     # 实例化 UAG：两层、窗口半径 6（窗口长度 13）、每步保留 8 个邻居
#     model = LocalUncertaintyAwareGraphAttentionLite(
#     dim_in=2048,          # x 的通道维 Cx
#     place_dim=2048,        # place 的通道维 Cp；若不用 place，置 0
#     num_heads=8,         # 需满足 (dim_in + place_dim) % num_heads == 0
#     window_size=9,       # 建议奇数：5/9/17
#     sim_temperature=0.07,
#     neighbor_temp=0.5,
#     uncertainty_mode="variance",  # 或 "confidence"
#     use_relative_pos_bias=True,
#     norm="ln",           # "ln" 更稳，"bn" 需要更大 batch
# ).to(f"cuda:{GPU}")
#
#     train_dataLoader = load_data(LABEL_PATH, IMG_PATH, PLC_PATH, MAMBA_PATH, SPLIT_PATH,
#                                  BATCH_SIZE, seg_sz=T, mode1='train', mode2=None)
#     test_dataLoader = load_data(LABEL_PATH, IMG_PATH, PLC_PATH, MAMBA_PATH, SPLIT_PATH,
#                                 BATCH_SIZE, seg_sz=T, mode1='test', mode2=None)
#
#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=1e-4,
#         betas=(0.9, 0.98),
#         weight_decay=1e-4,
#     )
#     lr_sh = torch.optim.lr_scheduler.LambdaLR(
#         optimizer,
#         warmup_decay_cosine(len(train_dataLoader), len(train_dataLoader) * (EPOCHS - 1))
#     )
#
#     for i in range(EPOCHS):
#         train_epoch(trainload=train_dataLoader, model=model, opti=optimizer, lr_sh=lr_sh, gpu=GPU)
#         ckpt_path = os.path.join(CKPT_DIR, f'epoch_{i:03d}.pt')
#
#         ckpt = {
#             "epoch": i,
#             "model_state_dict": model.state_dict(),
#             "optimizer_state_dict": optimizer.state_dict(),
#             "lr_scheduler_state_dict": lr_sh.state_dict() if lr_sh is not None else None,
#             "config": {
#                 "EPOCHS": EPOCHS,
#                 "BATCH_SIZE": BATCH_SIZE,
#                 "T": T,
#                 "GPU": GPU,
#                 "model_hparams": {
#                     "dim_in": 2048,
#                     "place_dim": 2048,
#                     "num_heads": 8,
#                     "window_size": 9,
#                     "sim_temperature": 0.07,
#                     "neighbor_temp": 0.5,
#                     "uncertainty_mode": "variance",
#                     "use_relative_pos_bias": True,
#                     "norm": "ln",
#                 },
#             },
#         }
#         torch.save(ckpt, ckpt_path)
#
#         side_path = os.path.join(SIDE_DIR, f'epoch_{i:03d}.npz')
#         met, moviePL = test_epoch(testload=test_dataLoader, model=model, gpu=GPU, side_save_path=side_path)
#         print(met)
#
#
# if __name__ == '__main__':
#     main()
#

import os
import glob
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tool.warmup_lr import warmup_decay_cosine
from tool.metric_scene import metric
from tool.loss import scene_boundary_detection_loss
from dataset.load_movienet_server import load_data
from model.RelationNet import LocalUncertaintyAwareGraphAttentionLite

# 配置
EPOCHS = 5
BATCH_SIZE = 128
T = 21
GPU = 5

# 损失函数配置
USE_FTL = True# 是否使用特征变换损失
USE_TCL = True  # 是否使用时序一致性损失
LAMBDA_FTL = 1  # FTL权重
LAMBDA_TCL = 1  # TCL权重
BETA = 5.0  # FTL中正样本的权重

# 路径
IMG_PATH = '/data/shared_dataset/shared_dataset/MovieDatasets/MovieNet/ImageNet_shot.pkl'
PLC_PATH = '/data/shared_dataset/shared_dataset/MovieDatasets/MovieNet/Places_shot.pkl'
LABEL_PATH = '/home/tianxiaoxuan/data/mamba/data/Movienet/label_endShot.pkl'
SPLIT_PATH = '/home/tianxiaoxuan/data/mamba/data/Movienet/split318.json'


CKPT_DIR = '/home/tianxiaoxuan/data/mamba/task1/checkpoint_scene_new'
SIDE_DIR = '/home/tianxiaoxuan/data/mamba/side_outputs'
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SIDE_DIR, exist_ok=True)


def train_epoch(trainload, model, opti, lr_sh, gpu=GPU):
    """
    训练一个epoch，使用综合损失函数
    """
    model.train()
    progress = tqdm(trainload)
    total_loss = 0.0
    loss_components = {'bce': 0.0, 'ftl': 0.0, 'tcl': 0.0}
    num_batches = 0

    for i, sample in enumerate(progress):
        name, img_ctx, plc_ctx, label, pos, _, _ = sample
        opti.zero_grad()

        img_ctx = img_ctx.cuda(gpu)
        plc_ctx = plc_ctx.cuda(gpu)
        label = label.cuda(gpu)

        # 前向传播
        model_output = model(img_ctx, plc_ctx)

        # 处理模型输出
        if isinstance(model_output, dict):
            # 模型返回字典：包含主输出和辅助输出
            pred = model_output.get('pred', model_output.get('main_pred'))
            r_pred = model_output.get('r_pred', None)  # 特征变换预测
            y_next = model_output.get('y_next', None)  # 下一时刻预测
        else:
            # 模型返回张量：仅主输出
            pred = model_output
            r_pred = None
            y_next = None

        pred = pred.squeeze()

        # 计算综合损失
        loss, loss_dict = scene_boundary_detection_loss(
            pred=pred,
            label=label.float(),
            r_pred=r_pred,
            y_next=y_next,
            use_ftl=USE_FTL and (r_pred is not None),
            use_tcl=USE_TCL and (y_next is not None),
            lambda_ftl=LAMBDA_FTL,
            lambda_tcl=LAMBDA_TCL,
            beta=BETA
        )

        # 反向传播
        loss.backward()
        opti.step()
        lr_sh.step()

        # 统计
        total_loss += loss.item()
        loss_components['bce'] += loss_dict.get('bce', 0.0)
        loss_components['ftl'] += loss_dict.get('ftl', 0.0)
        loss_components['tcl'] += loss_dict.get('tcl', 0.0)
        num_batches += 1

        # 更新进度条
        avg_loss = total_loss / num_batches
        postfix_str = f'loss={avg_loss:.6f}'
        if USE_FTL and loss_components['ftl'] > 0:
            postfix_str += f" | ftl={loss_components['ftl'] / num_batches:.6f}"
        if USE_TCL and loss_components['tcl'] > 0:
            postfix_str += f" | tcl={loss_components['tcl'] / num_batches:.6f}"
        progress.set_postfix_str(postfix_str)

    return 1


def test_epoch(testload, model, gpu=GPU, side_save_path=None,name = None):
    """
    测试阶段评估模型
    """
    predlist, labelist, pathlist, idlist = [], [], [], []
    side_rel_feat = []

    model.eval()
    with torch.no_grad():
        for i, sample in enumerate(tqdm(testload)):
            names, img_ctx, plc_ctx, label, pos, ids, ind = sample
            img_ctx = img_ctx.cuda(gpu)
            plc_ctx = plc_ctx.cuda(gpu)
            labelist.append(label.numpy())

            # 前向传播
            model_output = model(img_ctx, plc_ctx)

            # 处理模型输出
            if isinstance(model_output, dict):
                pred = model_output.get('pred', model_output.get('main_pred'))
            else:
                pred = model_output

            pred = pred.squeeze()
            predlist.append(pred.cpu().numpy())
            pathlist.append(names)
            idlist.append(np.asarray(ind))

    # 计算指标
    met, moviePL = metric(pathlist, predlist, labelist,name=name)

    if side_save_path is not None:
        labels_arr = np.concatenate(labelist)
        probs_arr = np.concatenate(predlist)
        ids_arr = np.concatenate(idlist)
        rel_feat_arr = np.concatenate(side_rel_feat) if len(side_rel_feat) else np.array([])
        paths = [p for batch_names in pathlist for p in batch_names]
        np.savez(side_save_path,
                 labels=labels_arr.astype(np.float32),
                 probs=probs_arr.astype(np.float32),
                 rel_feat=rel_feat_arr.astype(np.float32),
                 paths=np.array(paths),
                 ids=ids_arr.astype(np.int64))

    return met, moviePL




def main():
    """
    主训练函数
    """
    # 实例化模型：局部不确定性感知图注意力网络
    model = LocalUncertaintyAwareGraphAttentionLite(
        dim_in=2048,  # x 的通道维 Cx
        place_dim=2048,  # place 的通道维 Cp；若不用 place，置 0
        num_heads=8,  # 需满足 (dim_in + place_dim) % num_heads == 0
        window_size=9,  # 建议奇数：5/9/17
        sim_temperature=0.07,
        neighbor_temp=0.5,
        uncertainty_mode="variance",  # 或 "confidence"
        use_relative_pos_bias=True,
        norm="ln",  # "ln" 更稳，"bn" 需要更大 batch
    ).to(f"cuda:{GPU}")

    # 加载数据
    train_dataLoader = load_data(
        LABEL_PATH, IMG_PATH, PLC_PATH,  SPLIT_PATH,
        BATCH_SIZE, seg_sz=T, mode1='train', mode2=None
    )
    test_dataLoader = load_data(
        LABEL_PATH, IMG_PATH, PLC_PATH, SPLIT_PATH,
        BATCH_SIZE, seg_sz=T, mode1='test', mode2=None
    )

    # 优化器与学习率调度器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.98),
        weight_decay=1e-4,
    )
    lr_sh = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        warmup_decay_cosine(len(train_dataLoader), len(train_dataLoader) * (EPOCHS - 1))
    )

    # 打印训练配置
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Use FTL (Feature Transformation Loss): {USE_FTL}")
    print(f"Use TCL (Temporal Consistency Loss): {USE_TCL}")
    if USE_FTL:
        print(f"  - Lambda FTL: {LAMBDA_FTL}")
        print(f"  - Beta: {BETA}")
    if USE_TCL:
        print(f"  - Lambda TCL: {LAMBDA_TCL}")
    print("=" * 60 + "\n")

    # 训练循环
    for i in range(EPOCHS):
        print(f"\nEpoch {i + 1}/{EPOCHS}")
        train_epoch(trainload=train_dataLoader, model=model, opti=optimizer, lr_sh=lr_sh, gpu=GPU)

        # 保存检查点
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
                "BETA": BETA,
                "model_hparams": {
                    "dim_in": 2048,
                    "place_dim": 2048,
                    "num_heads": 8,
                    "window_size": 9,
                    "sim_temperature": 0.07,
                    "neighbor_temp": 0.5,
                    "uncertainty_mode": "variance",
                    "use_relative_pos_bias": True,
                    "norm": "ln",
                },
            },
        }
        torch.save(ckpt, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

        # # 测试评估
        side_path = os.path.join(SIDE_DIR, f'epoch_{i:03d}.npz')
        met, moviePL = test_epoch(testload=test_dataLoader, model=model, gpu=GPU, side_save_path=side_path, name=None)
        print(met)


if __name__ == '__main__':
    main()

