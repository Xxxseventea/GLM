
"""
scene2event_schemeA_fixed.py
-----------------------------
终身学习 (Scene→Event)
- 新域: Event (Kinetics)
- 旧域: Scene (MovieNet)
包含: Teacher(冻结Scene), Student(Event head), Feature Distillation + Domain Adversarial
"""

import os
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np

from model.backbone.RelationNet import LocalUncertaintyAwareGraphAttentionLite
from model.detector.scene_detector import MlpHead
from model.detector.event_detector import FrameWiseHead
from model.discriminator.discriminator import TemporalDiscriminator
from dataset.movienet.load_movienet_server import load_data
from dataset.kinetics.dataset import build as build_kinetics, collate_fn
from tool.warmup_lr import warmup_decay_cosine

# ====================
# Config
# ====================
DEVICE = "cuda:0"
EPOCHS = 20
BATCH_SIZE = 128
T = 21
ALPHA_FEAT = 0.5
LAMBDA_ADV = 0.1
LR = 1e-4
EVENT_KEEP_RATIO = 0.7
SCENE_KEEP_RATIO = 0.3

CKPT_SCENE = "/home/tianxiaoxuan/data/mamba/checkpoint_scene/epoch_004.pt"
OUT_DIR = "/home/tianxiaoxuan/data/mamba/checkpoint_sceneWithCL"
os.makedirs(OUT_DIR, exist_ok=True)
from tool.metric.scene_metric_backup import metric as scene_metric
# Scene dataset
IMG_PATH = '/data/shared_dataset/shared_dataset/MovieDatasets/MovieNet/ImageNet_shot.pkl'
PLC_PATH = '/data/shared_dataset/shared_dataset/MovieDatasets/MovieNet/Places_shot.pkl'
LABEL_PATH = '/home/tianxiaoxuan/data/mamba/data/label_endShot.pkl'
SPLIT_PATH_SCENE = '/home/tianxiaoxuan/data/mamba/data/split318.json'
MAMBA_PATH = '/home/tianxiaoxuan/data/mamba/data/ImageNet_shot.pkl'

# Event dataset
KINETICS_PATH = {
    "feature_path": "/data/shared_dataset/Kinetics/features",
    "annotation_path": "/data/shared_dataset/Kinetics/data",
    "score_path": "/data/shared_dataset/Kinetics/data",
}


# ============== EncoderHook ==============
class EncoderPreHook:
    def __init__(self, model_with_head: nn.Module):
        self.z = None
        def pre_hook(module, inputs):
            z = inputs[0]
            self.z = z.mean(dim=1) if z.dim() == 3 else z
        self.h = model_with_head.detector.register_forward_pre_hook(lambda m, i: pre_hook(m, i))
    def close(self):
        self.h.remove()


# ============== Data ==============
def build_subset_loader_event(keep_ratio, batch_size, seg_sz, device):
    args = argparse.Namespace()
    args.feature_path = KINETICS_PATH["feature_path"]
    args.annotation_path = KINETICS_PATH["annotation_path"]
    args.score_path = KINETICS_PATH["score_path"]
    args.window_size = seg_sz
    args.interval = 1
    args.device = device
    dataset_full = build_kinetics(split='train', args=args)
    N = len(dataset_full)
    keep = max(1, int(N * keep_ratio))
    subset = Subset(dataset_full, torch.randperm(N)[:keep].tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True)
    print(f"[Event] subset {keep}/{N} ({keep/N:.1%})")
    return loader


def build_subset_loader_scene(split_json, keep_ratio, batch_size, seg_sz):
    full_loader = load_data(LABEL_PATH, IMG_PATH, PLC_PATH,  split_json,
                            batch_size, seg_sz=seg_sz, mode1='train', mode2=None)
    dataset = full_loader.dataset
    N = len(dataset)
    keep = max(1, int(N * keep_ratio))
    subset = Subset(dataset, torch.randperm(N)[:keep].tolist())
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    print(f"[Scene] subset {keep}/{N} ({keep/N:.1%})")
    return loader


# ============== Teacher/Student ==============
def build_teacher(scene_ckpt: str, device=DEVICE):
    teacher = LocalUncertaintyAwareGraphAttentionLite(
        dim_in=2048, num_heads=8, window_size=5, sim_temperature=0.07,
        neighbor_temp=0.5, uncertainty_mode="variance", use_relative_pos_bias=True,
        norm="ln", size=T).to(device)
    ckpt = torch.load(scene_ckpt, map_location="cpu")
    sd = ckpt.get("model", ckpt)
    pruned = {k: v for k, v in sd.items() if not k.startswith("detector.")}
    teacher.load_state_dict(pruned, strict=False)
    teacher.detector = MlpHead(in_dim=2176, hid_dim=512, out_dim=1).to(device)
    for p in teacher.parameters(): p.requires_grad_(False)
    teacher.eval()
    hook = EncoderPreHook(teacher)
    print("🧑‍🏫 Teacher(Scene) loaded.")
    return teacher, hook, 2176


def build_student(device=DEVICE):
    student = LocalUncertaintyAwareGraphAttentionLite(
        dim_in=2048, num_heads=8, window_size=5, sim_temperature=0.07,
        neighbor_temp=0.5, uncertainty_mode="variance", use_relative_pos_bias=True,
        norm="ln", size=T).to(device)
    student.detector = FrameWiseHead(in_features=2176).to(device)
    hook = EncoderPreHook(student)
    print("🧠 Student(Event head) initialized.")
    return student, hook, 2176


# ============== GRL ==============
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


# ============== Training ==============
def paired_iter(loader_a, loader_b):
    ita, itb = iter(loader_a), iter(loader_b)
    while True:
        try:
            a = next(ita)
        except StopIteration:
            ita = iter(loader_a); a = next(ita)
        try:
            b = next(itb)
        except StopIteration:
            itb = iter(loader_b); b = next(itb)
        yield a, b


# ===============================================================
# Scene Head 训练 & 评估
# ===============================================================
def train_one_epoch(scene_loader, model, optimizer, scheduler, criterion, gpu=0, epoch=1):
    """scene_head 单次训练(一轮)"""
    model.train().cuda(gpu)
    epoch_loss = 0.0
    for batch_idx, batch in enumerate(tqdm(scene_loader, desc=f"🎯 Train Scene Epoch {epoch}")):
        names, img_ctx, plc_ctx, label, pos, ids, ind = batch = batch
        img_ctx, plc_ctx, label = img_ctx.cuda(gpu), plc_ctx.cuda(gpu), label.float().cuda(gpu)

        optimizer.zero_grad()
        x = 0.7 * img_ctx + 0.3 * plc_ctx
        logtics, pred = model(x)
        loss = criterion(logtics.squeeze(), label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(scene_loader)
    lr = scheduler.get_last_lr()[0]
    print(f"[Scene Train] Epoch {epoch}: Loss={avg_loss:.4f}, LR={lr:.6f}")
    return avg_loss


@torch.no_grad()
def evaluate_scene(scene_loader, model, gpu=0):
    """Scene Head 推理并计算 metric"""
    model.eval().cuda(gpu)
    predlist, labelist, pathlist = [], [], []
    for batch in tqdm(scene_loader, desc="Scene Eval"):
        names, img_ctx, plc_ctx, label, pos, ids, ind = batch = batch
        img_ctx, plc_ctx = img_ctx.cuda(gpu), plc_ctx.cuda(gpu)
        x = 0.7 * img_ctx + 0.3 * plc_ctx
        _, pred = model(x)
        predlist.append(pred.cpu().numpy())
        labelist.append(label.numpy())
        pathlist.append(names)
    met, moviePL = scene_metric(pathlist, predlist, labelist)
    return met
def train_epoch_schemeA(event_loader, scene_loader,
                        student, stu_hook, teacher, tea_hook, disc,
                        opt_main, opt_disc, sch_main=None, sch_disc=None,
                        device=DEVICE, alpha_feat=ALPHA_FEAT, lambda_adv=LAMBDA_ADV):

    student.train().to(device)
    teacher.eval().to(device)
    disc.train().to(device)
    bce = nn.BCEWithLogitsLoss()

    it = paired_iter(event_loader, scene_loader)
    pbar = tqdm(range(len(event_loader)), desc="Train SchemeA(Event70 + Scene30)")

    for _ in pbar:
        batch_event, batch_scene = next(it)
        # ----- Event batch (new domain)
        locations, features, targets_list, _, _, _ = batch_event
        features = features.to(device)
        label_e = torch.tensor([1.0 if t["boundaries"].numel() > 0 else 0.0
                                for t in targets_list], dtype=torch.float32, device=device)

        # ----- Scene batch (old domain)
        _, img_s, plc_s,  label_s, _, _, _ = batch_scene
        img_s, plc_s = img_s.to(device), plc_s.to(device)
        w = 0.7
        x_old = w * img_s + (1 - w) * plc_s
        x_new = features

        # (1) 更新判别器
        opt_disc.zero_grad(set_to_none=True)
        with torch.no_grad():
            _ = student(x_new); z_new = stu_hook.z.detach()
            _ = student(x_old); z_old = stu_hook.z.detach()
        z_dom = torch.cat([z_old, z_new], dim=0)
        y_dom = torch.cat([torch.zeros(z_old.size(0),1,device=device),
                           torch.ones(z_new.size(0),1,device=device)], 0)
        logit_d = disc(z_dom)
        loss_disc = F.binary_cross_entropy_with_logits(logit_d, y_dom)
        loss_disc.backward(); opt_disc.step()
        if sch_disc is not None: sch_disc.step()

        # (2) 更新学生
        opt_main.zero_grad(set_to_none=True)
        sc_new, _ = student(x_new)
        loss_task = bce(sc_new.squeeze(), label_e)

        with torch.no_grad():
            _ = teacher(x_new); z_t = tea_hook.z
        _ = student(x_new); z_s = stu_hook.z
        loss_feat = F.mse_loss(z_s, z_t)

        _ = student(x_old); z_old_s = stu_hook.z
        z_all = torch.cat([z_old_s, z_s], dim=0)
        y_all = torch.cat([torch.zeros(z_old_s.size(0),1,device=device),
                           torch.ones(z_s.size(0),1,device=device)],0)
        logits_adv = disc(GradReverse.apply(z_all, 1.0))
        loss_adv = F.binary_cross_entropy_with_logits(logits_adv, y_all)

        loss_total = loss_task + alpha_feat * loss_feat - lambda_adv * loss_adv
        loss_total.backward(); opt_main.step()
        if sch_main is not None: sch_main.step()

        pbar.set_postfix(task=loss_task.item(), feat=loss_feat.item(),
                         adv=loss_adv.item(), disc=loss_disc.item())


# ============== main ==============
def main():
    teacher, tea_hook, tea_dim = build_teacher(CKPT_SCENE, DEVICE)
    student, stu_hook, stu_dim = build_student(DEVICE)
    disc = TemporalDiscriminator(in_dim=stu_dim).to(DEVICE)

    loader_event = build_subset_loader_event(EVENT_KEEP_RATIO, BATCH_SIZE, T, DEVICE)
    loader_scene = build_subset_loader_scene(SPLIT_PATH_SCENE, SCENE_KEEP_RATIO, BATCH_SIZE, T)

    opt_main = torch.optim.Adam(student.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=1e-4)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=LR, betas=(0.9, 0.98), weight_decay=1e-4)
    sch_main = torch.optim.lr_scheduler.LambdaLR(
        opt_main, warmup_decay_cosine(len(loader_event), len(loader_event)*(EPOCHS-1)))
    sch_disc = torch.optim.lr_scheduler.LambdaLR(
        opt_disc, warmup_decay_cosine(len(loader_event), len(loader_event)*(EPOCHS-1)))



    train_epoch_schemeA(loader_event, loader_scene,
                        student, stu_hook, teacher, tea_hook, disc,
                        opt_main, opt_disc, sch_main, sch_disc, DEVICE)
    torch.save({"model": student.state_dict(), "epoch": 0},
               os.path.join(OUT_DIR, f"student_ep0.pth"))
    # =======================================================
    # 2️⃣ Scene Head fine-tune & evaluate each epoch
    # =======================================================
    print("\n🎞 Fine-tuning Scene Head ...")
    gpu = 5

    scene_head = MlpHead(in_dim=2176, hid_dim=512).to(f"cuda:{gpu}")
    for n, p in student.named_parameters():
        p.requires_grad = False
    student.detector = scene_head
    for p in student.detector.parameters():
        p.requires_grad = True

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(student.detector.parameters(), lr=1e-4,
                                 betas=(0.9, 0.98), weight_decay=1e-4)
    total_steps = len(loader_scene) * EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_decay_cosine(len(loader_scene), total_steps))

    for ep in range(1, EPOCHS + 1):
        train_one_epoch(loader_scene, student, optimizer, scheduler, criterion, gpu=gpu, epoch=ep)
        scene_metrics = evaluate_scene(loader_scene, student, gpu=gpu)
        print(f"\n[Scene Eval after Epoch {ep}] Metrics: {scene_metrics}")

    print("\n✅ Training + Evaluation Pipeline Completed.")

    print("✅ Scene→Event SchemeA training finished.")


if __name__ == "__main__":
    main()


#
# """
# scene2event_schemeB_prehook.py
# -------------------------------
# 终身学习 (Scene → Event)
# Scheme B：Event + Scene 联合训练，包含对抗与蒸馏
# 采用 PreHook（与 Scheme A 相同）方式提取特征，保持判别器维度稳定
# """
# import os, torch, argparse
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Subset
# from tqdm import tqdm
# from model.backbone.RelationNet import LocalUncertaintyAwareGraphAttentionLite
# from model.detector.scene_detector import MlpHead
# from model.detector.event_detector import FrameWiseHead
# from model.discriminator.discriminator import TemporalDiscriminator
# from dataset.movienet.load_movienet_server import load_data
# from dataset.kinetics.dataset import build as build_kinetics, collate_fn
# from tool.warmup_lr import warmup_decay_cosine
#
# # ================= Config =================
# DEVICE           = "cuda:5"
# EPOCHS           = 10
# BATCH_SIZE       = 128
# T                = 21
# LR               = 1e-4
# ALPHA_FEAT       = 0.5
# LAMBDA_ADV       = 0.1
# EVENT_KEEP_RATIO = 0.6
# SCENE_KEEP_RATIO = 0.4
#
# CKPT_SCENE = "/home/tianxiaoxuan/data/mamba/checkpoint_scene/epoch_004.pt"
# OUT_DIR    = "/home/tianxiaoxuan/data/mamba/checkpoint_sceneWithCL"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# IMG_PATH  = '/data/shared_dataset/shared_dataset/MovieDatasets/MovieNet/ImageNet_shot.pkl'
# PLC_PATH  = '/data/shared_dataset/shared_dataset/MovieDatasets/MovieNet/Places_shot.pkl'
# LABEL_PATH = '/home/tianxiaoxuan/data/mamba/data/label_endShot.pkl'
# SPLIT_PATH_SCENE = '/home/tianxiaoxuan/data/mamba/data/split318.json'
# MAMBA_PATH = '/home/tianxiaoxuan/data/mamba/data/ImageNet_shot.pkl'
# KINETICS_PATH = {
#     "feature_path": "/data/shared_dataset/Kinetics/features",
#     "annotation_path": "/data/shared_dataset/Kinetics/data",
#     "score_path": "/data/shared_dataset/Kinetics/data",
# }
#
# # ============== EncoderPreHook（与 Scheme A 相同） ==============
# class EncoderPreHook:
#     def __init__(self, model_with_head: nn.Module):
#         self.z = None
#         def pre_hook(module, inputs):
#             z = inputs[0]
#             self.z = z.mean(dim=1) if z.dim() == 3 else z
#         self.h = model_with_head.detector.register_forward_pre_hook(lambda m, i: pre_hook(m, i))
#     def close(self):
#         self.h.remove()
#
# # ============== 数据加载 ==============
# def build_subset_loader_event(keep_ratio,batch_size,seg_sz,device):
#     args = argparse.Namespace(
#         feature_path=KINETICS_PATH["feature_path"],
#         annotation_path=KINETICS_PATH["annotation_path"],
#         score_path=KINETICS_PATH["score_path"],
#         window_size=seg_sz, interval=1, device=device)
#     dataset_full = build_kinetics(split='train', args=args)
#     N=len(dataset_full); keep=max(1,int(N*keep_ratio))
#     subset=Subset(dataset_full,torch.randperm(N)[:keep].tolist())
#     loader=DataLoader(subset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn,
#                       num_workers=4,pin_memory=True,drop_last=True)
#     print(f"[Event] subset {keep}/{N} ({keep/N:.1%})")
#     return loader
#
# def build_subset_loader_scene(split_json,keep_ratio,batch_size,seg_sz):
#     full_loader=load_data(LABEL_PATH,IMG_PATH,PLC_PATH,MAMBA_PATH,split_json,
#                           batch_size,seg_sz=seg_sz,mode1='train',mode2=None)
#     dataset=full_loader.dataset; N=len(dataset)
#     keep=max(1,int(N*keep_ratio))
#     subset=Subset(dataset,torch.randperm(N)[:keep].tolist())
#     loader=DataLoader(subset,batch_size=batch_size,shuffle=True,num_workers=4,
#                       pin_memory=True,drop_last=True)
#     print(f"[Scene] subset {keep}/{N} ({keep/N:.1%})")
#     return loader
#
# # ============== Teacher / Student ==============
# def build_teacher(scene_ckpt:str,device=DEVICE):
#     teacher=LocalUncertaintyAwareGraphAttentionLite(
#         dim_in=2048,num_heads=8,window_size=5,sim_temperature=0.07,
#         neighbor_temp=0.5,uncertainty_mode="variance",use_relative_pos_bias=True,
#         norm="ln",size=T).to(device)
#     ckpt=torch.load(scene_ckpt,map_location="cpu"); sd=ckpt.get("model",ckpt)
#     pruned={k:v for k,v in sd.items() if not k.startswith("detector.")}
#     teacher.load_state_dict(pruned,strict=False)
#     teacher.detector=MlpHead(in_dim=2176,hid_dim=512,out_dim=1).to(device)
#     for p in teacher.parameters(): p.requires_grad_(False)
#     teacher.eval()
#     hook=EncoderPreHook(teacher)
#     return teacher,hook,2176
#
# def build_student(device=DEVICE):
#     student=LocalUncertaintyAwareGraphAttentionLite(
#         dim_in=2048,num_heads=8,window_size=5,sim_temperature=0.07,
#         neighbor_temp=0.5,uncertainty_mode="variance",use_relative_pos_bias=True,
#         norm="ln",size=T).to(device)
#     student.detector=FrameWiseHead(in_features=2176).to(device)
#     hook=EncoderPreHook(student)
#     return student,hook,2176
#
# # ============== 反向梯度层 ==============
# class GradReverse(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx,x,lambd):
#         ctx.lambd=lambd; return x.view_as(x)
#     @staticmethod
#     def backward(ctx,grad_output):
#         return -ctx.lambd*grad_output,None
#
# # ============== Data pairing generator ==============
# def paired_iter(loader_a,loader_b):
#     ita,itb=iter(loader_a),iter(loader_b)
#     while True:
#         try: a=next(ita)
#         except StopIteration: ita=iter(loader_a); a=next(ita)
#         try: b=next(itb)
#         except StopIteration: itb=iter(loader_b); b=next(itb)
#         yield a,b
#
# # ============== 训练逻辑 Scheme B（PreHook） ==============
# def train_epoch_schemeB(loader_event,loader_scene,
#                         student,stu_hook,teacher,tea_hook,disc,
#                         opt_main,opt_disc,sch_main=None,sch_disc=None,
#                         device=DEVICE,alpha_feat=ALPHA_FEAT,lambda_adv=LAMBDA_ADV):
#
#     student.train(); teacher.eval(); disc.train()
#     bce=nn.BCEWithLogitsLoss()
#     it=paired_iter(loader_event,loader_scene)
#     pbar=tqdm(range(len(loader_event)),desc="Scheme B(Event+Scene)")
#
#     for _ in pbar:
#         batch_event,batch_scene=next(it)
#
#         # ==== Event domain ====
#         locations,features,targets_list,_,_,_=batch_event
#         features=features.to(device)
#         label_e=torch.tensor([1. if t["boundaries"].numel()>0 else 0.
#                               for t in targets_list],dtype=torch.float32,device=device)
#
#         # ==== Scene domain ====
#         _,img_s,plc_s,_,label_s,_,_,_=batch_scene
#         img_s,plc_s=img_s.to(device),plc_s.to(device)
#         x_s=0.7*img_s+0.3*plc_s
#         x_e=features
#
#         # ===== (1) 判别器更新 =====
#         opt_disc.zero_grad(set_to_none=True)
#         with torch.no_grad():
#             _=student(x_e); z_e=stu_hook.z.detach()
#             _=student(x_s); z_s=stu_hook.z.detach()
#         z_dom=torch.cat([z_s,z_e],dim=0)
#         y_dom=torch.cat([torch.zeros(z_s.size(0),1,device=device),
#                          torch.ones(z_e.size(0),1,device=device)],0)
#         out_d=disc(z_dom)
#         loss_disc=F.binary_cross_entropy_with_logits(out_d,y_dom)
#         loss_disc.backward(); opt_disc.step()
#         if sch_disc: sch_disc.step()
#
#         # ===== (2) 学生更新 =====
#         opt_main.zero_grad(set_to_none=True)
#         sc_e,_=student(x_e)
#         loss_task=bce(sc_e.squeeze(),label_e)
#
#         # Distillation loss
#         with torch.no_grad():
#             _=teacher(x_e); z_t=tea_hook.z
#         _=student(x_e); z_s_new=stu_hook.z
#         loss_feat=F.mse_loss(z_s_new,z_t)
#
#         # Adversarial Loss (GRL)
#         _=student(x_s); z_s_old=stu_hook.z
#         z_all=torch.cat([z_s_old,z_s_new],dim=0)
#         y_all=torch.cat([torch.zeros(z_s_old.size(0),1,device=device),
#                          torch.ones(z_s_new.size(0),1,device=device)],0)
#         logits_adv=disc(GradReverse.apply(z_all,1.0))
#         loss_adv=F.binary_cross_entropy_with_logits(logits_adv,y_all)
#
#         loss_total=loss_task+alpha_feat*loss_feat-lambda_adv*loss_adv
#         loss_total.backward(); opt_main.step()
#         if sch_main: sch_main.step()
#
#         pbar.set_postfix(task=loss_task.item(),feat=loss_feat.item(),
#                          adv=loss_adv.item(),disc=loss_disc.item())
#
# # ============== main ==============
# def main():
#     teacher,tea_hook,dim_t=build_teacher(CKPT_SCENE,DEVICE)
#     student,stu_hook,dim_s=build_student(DEVICE)
#     disc=TemporalDiscriminator(in_dim=dim_s).to(DEVICE)
#
#     loader_event=build_subset_loader_event(EVENT_KEEP_RATIO,BATCH_SIZE,T,DEVICE)
#     loader_scene=build_subset_loader_scene(SPLIT_PATH_SCENE,SCENE_KEEP_RATIO,BATCH_SIZE,T)
#
#     opt_main=torch.optim.Adam(student.parameters(),lr=LR,betas=(0.9,0.98),weight_decay=1e-4)
#     opt_disc=torch.optim.Adam(disc.parameters(),lr=LR,betas=(0.9,0.98),weight_decay=1e-4)
#     sch_main=torch.optim.lr_scheduler.LambdaLR(opt_main,
#         warmup_decay_cosine(len(loader_event),len(loader_event)*(EPOCHS-1)))
#     sch_disc=torch.optim.lr_scheduler.LambdaLR(opt_disc,
#         warmup_decay_cosine(len(loader_event),len(loader_event)*(EPOCHS-1)))
#
#     for ep in range(EPOCHS):
#         print(f"\n===== Epoch {ep}/{EPOCHS} =====")
#         train_epoch_schemeB(loader_event,loader_scene,student,stu_hook,
#                             teacher,tea_hook,disc,opt_main,opt_disc,
#                             sch_main,sch_disc,DEVICE)
#         torch.save({"model":student.state_dict(),"epoch":ep},
#                    os.path.join(OUT_DIR,f"student_ep{ep:03d}.pth"))
#     print("✅ Scene→Event Scheme B PreHook版训练完成")
#
# if __name__ == "__main__":
#     main()

#
# # scene2event_train.py
# import os, torch, argparse
# from tqdm import tqdm
# from torch import nn
# from torch.utils.data import DataLoader, Subset
# from collections import defaultdict
# import torch.nn.functional as F
#
# # ======= Dependences =======
# from model.backbone.RelationNet import LocalUncertaintyAwareGraphAttentionLite
# from model.detector.event_detector import FrameWiseHead
# from model.discriminator.discriminator import TemporalDiscriminator
# from dataset.movienet.load_movienet_server import load_data
# from dataset.kinetics.dataset import build as build_kinetics, collate_fn
# from tool.warmup_lr import warmup_decay_cosine
#
# # ======= Config =======
# DEVICE = "cuda:5"
# GPU = 5
# EPOCHS = 10
# BATCH_SIZE = 128
# T = 21
# LR = 1e-4
# ALPHA_FEAT = 0.5
# LAMBDA_ADV = 0.1
# EVENT_KEEP_RATIO = 0.7
# SCENE_KEEP_RATIO = 0.3
#
# OUT_DIR = "/home/tianxiaoxuan/data/mamba/checkpoint_sceneWithCL"
# os.makedirs(OUT_DIR, exist_ok=True)
# CKPT_SCENE = "/home/tianxiaoxuan/data/mamba/checkpoint_scene/epoch_004.pt"
#
# IMG_PATH = '/data/shared_dataset/shared_dataset/MovieDatasets/MovieNet/ImageNet_shot.pkl'
# PLC_PATH = '/data/shared_dataset/shared_dataset/MovieDatasets/MovieNet/Places_shot.pkl'
# LABEL_PATH = '/home/tianxiaoxuan/data/mamba/data/label_endShot.pkl'
# SPLIT_PATH_SCENE = '/home/tianxiaoxuan/data/mamba/data/split318.json'
# MAMBA_PATH = '/home/tianxiaoxuan/data/mamba/data/ImageNet_shot.pkl'
# KINETICS_PATH = {
#     "feature_path": "/data/shared_dataset/Kinetics/features",
#     "annotation_path": "/data/shared_dataset/Kinetics/data",
#     "score_path": "/data/shared_dataset/Kinetics/data",
# }
#
# # ======= Load utils =======
# def extract_state_dict(ckpt):
#     if isinstance(ckpt, dict):
#         for key in ("model_state_dict", "state_dict", "model"):
#             if key in ckpt:
#                 return ckpt[key]
#     return ckpt
#
# def strip_module_prefix(sd: dict):
#     return {k[len("module."):] if k.startswith("module.") else k:v for k,v in sd.items()}
#
# def load_backbone_only(model: nn.Module, ckpt_path: str, detector_prefix="detector."):
#     ckpt = torch.load(ckpt_path, map_location="cpu")
#     sd = strip_module_prefix(extract_state_dict(ckpt))
#     pruned = {k:v for k,v in sd.items() if not k.startswith(detector_prefix)}
#     res = model.load_state_dict(sd, strict=False)
#     print("Backbone loaded:", len(pruned), "params.")
#     return res
#
# # ======= Hook =======
# class EncoderPreHook:
#     def __init__(self, model_with_head):
#         self.z = None
#         def pre_hook(m, inp):
#             z = inp[0]
#             self.z = z.mean(dim=1) if z.dim()==3 else z
#         self.h = model_with_head.detector.register_forward_pre_hook(lambda m,i: pre_hook(m,i))
#     def close(self): self.h.remove()
#
# # ======= Build models =======
# def build_teacher(scene_ckpt, device=DEVICE):
#     teacher = LocalUncertaintyAwareGraphAttentionLite(
#         dim_in=2048, num_heads=8, window_size=5, sim_temperature=0.07,
#         neighbor_temp=0.5, uncertainty_mode="variance",
#         use_relative_pos_bias=True, norm="ln", size=T
#     ).to(device)
#     load_backbone_only(teacher, scene_ckpt)
#     for p in teacher.parameters(): p.requires_grad_(False)
#     teacher.eval()
#     teacher.detector = nn.Identity().to(device)
#     hook = EncoderPreHook(teacher)
#     print("Teacher(Scene) encoder loaded.")
#     return teacher, hook, 2176
#
# def build_student(device=DEVICE):
#     student = LocalUncertaintyAwareGraphAttentionLite(
#         dim_in=2048, num_heads=8, window_size=5, sim_temperature=0.07,
#         neighbor_temp=0.5, uncertainty_mode="variance",
#         use_relative_pos_bias=True, norm="ln", size=T
#     ).to(device)
#     student.detector = FrameWiseHead(in_features=2176).to(device)
#     hook = EncoderPreHook(student)
#     print("Student(Event head) initialized.")
#     return student, hook, 2176
#
# # ======= Grad Reverse =======
# class GradReverse(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, lambd): ctx.lambd = lambd; return x.view_as(x)
#     @staticmethod
#     def backward(ctx, grad_output): return -ctx.lambd * grad_output, None
# def grad_reverse(x, lambd=1.0): return GradReverse.apply(x, lambd)
#
# # ======= Dataloaders =======
# def build_subset_loader_event(keep,batch,seg,device):
#     args = argparse.Namespace(feature_path=KINETICS_PATH["feature_path"],
#                               annotation_path=KINETICS_PATH["annotation_path"],
#                               score_path=KINETICS_PATH["score_path"],
#                               window_size=seg, interval=1, device=device)
#     ds = build_kinetics(split='train', args=args)
#     import random
#     N = len(ds); keep = max(1,int(N*keep))
#     sub = Subset(ds, torch.randperm(N)[:keep].tolist())
#     return DataLoader(sub,batch_size=batch,shuffle=True,collate_fn=collate_fn,
#                       num_workers=4,pin_memory=True,drop_last=True)
#
# def build_subset_loader_scene(split,keep,batch,seg):
#     full = load_data(LABEL_PATH, IMG_PATH, PLC_PATH, MAMBA_PATH, split,
#                      batch, seg_sz=seg, mode1='train', mode2=None)
#     ds = full.dataset; N = len(ds); keep = max(1,int(N*keep))
#     sub = Subset(ds, torch.randperm(N)[:keep].tolist())
#     return DataLoader(sub,batch_size=batch,shuffle=True,num_workers=4,
#                       pin_memory=True,drop_last=True)
#
# def paired_iter(a,b):
#     ia,ib=iter(a),iter(b)
#     while True:
#         try: A=next(ia)
#         except StopIteration: ia=iter(a); A=next(ia)
#         try: B=next(ib)
#         except StopIteration: ib=iter(b); B=next(ib)
#         yield A,B
#
# # ======= Train SchemeB =======
# def train_epoch_schemeB(loaderE, loaderS, student, stu_hook, teacher, tea_hook, disc,
#                         optM, optD, schM=None, schD=None,
#                         device=DEVICE, alpha_feat=ALPHA_FEAT, lambda_adv=LAMBDA_ADV):
#     bce = nn.BCEWithLogitsLoss()
#     it = paired_iter(loaderE, loaderS)
#     pbar = tqdm(range(len(loaderE)), desc="Train SchemeB Scene→Event")
#     for _ in pbar:
#         batchE, batchS = next(it)
#         loc, feat, tlist, *_ = batchE
#         feat = feat.to(device)
#         labelE = torch.tensor([1. if t['boundaries'].numel()>0 else 0. for t in tlist],
#                               device=device)
#
#         _, imgS, plcS, _, labelS, *_ = batchS
#         imgS, plcS = imgS.to(device), plcS.to(device)
#         x_old = 0.7*imgS + 0.3*plcS
#         x_new = feat
#
#         # 判别器
#         optD.zero_grad(set_to_none=True)
#         with torch.no_grad():
#             _ = student(x_new); zE = stu_hook.z.detach()
#             _ = student(x_old); zS = stu_hook.z.detach()
#         z_dom = torch.cat([zS, zE], 0)
#         y_dom = torch.cat([torch.zeros(len(zS),1,device=device),
#                            torch.ones(len(zE),1,device=device)], 0)
#         outD = disc(z_dom)
#         lossD = F.binary_cross_entropy_with_logits(outD, y_dom)
#         lossD.backward(); optD.step()
#         if schD: schD.step()
#
#         # 主模型
#         optM.zero_grad(set_to_none=True)
#         scoreE, _ = student(x_new)
#         loss_task = bce(scoreE.squeeze(), labelE)
#
#         with torch.no_grad():
#             _ = teacher(x_new); zT = tea_hook.z
#         _ = student(x_new); zS_new = stu_hook.z
#         loss_feat = F.mse_loss(zS_new, zT)
#
#         _ = student(x_old); zS_old = stu_hook.z
#         z_all = torch.cat([zS_old, zS_new], 0)
#         y_all = torch.cat([torch.zeros(len(zS_old),1,device=device),
#                            torch.ones(len(zS_new),1,device=device)], 0)
#         out_adv = disc(grad_reverse(z_all,1.0))
#         loss_adv = F.binary_cross_entropy_with_logits(out_adv, y_all)
#         #
#         # loss = loss_task + alpha_feat*loss_feat - lambda_adv*loss_adv
#         loss = loss_task + alpha_feat * loss_feat
#         loss.backward(); optM.step()
#         if schM: schM.step()
#
#         pbar.set_postfix(task=loss_task.item(), feat=loss_feat.item(),
#                          adv=loss_adv.item(), disc=lossD.item())
#
# def train_epoch_schemeA_mixed(loader_scene, loader_event,
#                               student, stu_hook,
#                               teacher, tea_hook,
#                               disc,
#                               opt_student, opt_disc,
#                               sch_student=None, sch_disc=None,
#                               device=DEVICE,
#                               alpha_feat=ALPHA_FEAT,
#                               lambda_adv=LAMBDA_ADV):
#     """
#     Scheme A: Scene(Event)混合批训练
#     - Scene: 有 label → 任务监督 + 蒸馏
#     - Event: 无 label → 仅对抗
#     """
#     student.train(); teacher.eval(); disc.train()
#     bce = nn.BCEWithLogitsLoss()
#
#     it = paired_iter(loader_scene, loader_event)
#     pbar = tqdm(range(len(loader_scene)), desc="Train SchemeA (Scene+Event mixed)")
#
#     for _ in pbar:
#         batchS, batchE = next(it)
#         # ------- Scene batch -------
#         _, imgS, plcS, _, labelS, *_ = batchS
#         imgS, plcS = imgS.to(device), plcS.to(device)
#         labelS = labelS.float().to(device)
#         x_scene = 0.7 * imgS + 0.3 * plcS
#
#         # ------- Event batch -------
#         loc, feat, tlist, *_ = batchE
#         feat = feat.to(device)
#
#         # ------- 更新判别器 -------
#         opt_disc.zero_grad(set_to_none=True)
#         with torch.no_grad():
#             _ = student(x_scene); z_scene = stu_hook.z.detach()
#             _ = student(feat);    z_event = stu_hook.z.detach()
#         z_all = torch.cat([z_event, z_scene], 0)
#         y_all = torch.cat([
#             torch.zeros(len(z_event), 1, device=device),
#             torch.ones(len(z_scene), 1, device=device)
#         ], 0)
#         outD = disc(z_all)
#         lossD = F.binary_cross_entropy_with_logits(outD, y_all)
#         lossD.backward(); opt_disc.step()
#         if sch_disc: sch_disc.step()
#
#         # ------- 更新学生 -------
#         opt_student.zero_grad(set_to_none=True)
#
#         # (1) Scene 任务监督
#         out_scene, _ = student(x_scene)
#         loss_task = bce(out_scene.squeeze(), labelS)
#
#         # (2) 特征蒸馏 (teacher(Scene) → student(Scene))
#         with torch.no_grad():
#             _ = teacher(x_scene); zT = tea_hook.z
#         _ = student(x_scene); zS = stu_hook.z
#         loss_feat = F.mse_loss(zS, zT)
#
#         # (3) 域对抗 (Scene vs Event)
#         _ = student(feat); z_event_s = stu_hook.z
#         z_domain = torch.cat([z_event_s, zS], 0)
#         y_domain = torch.cat([
#             torch.zeros(len(z_event_s), 1, device=device),
#             torch.ones(len(zS), 1, device=device)
#         ], 0)
#         out_adv = disc(grad_reverse(z_domain, 1.0))
#         loss_adv = F.binary_cross_entropy_with_logits(out_adv, y_domain)
#
#         # (4) 总损失 = 任务 + 蒸馏 − 对抗
#         # loss = loss_task + alpha_feat * loss_feat - lambda_adv * loss_adv
#         loss = loss_task + - lambda_adv * loss_adv
#         loss.backward(); opt_student.step()
#         if sch_student: sch_student.step()
#
#         pbar.set_postfix(task=loss_task.item(),
#                          feat=loss_feat.item(),
#                          adv=loss_adv.item(),
#                          disc=lossD.item())
#
# from model.detector.scene_detector import MlpHead
# # ======= MAIN =======
# def main():
#     teacher, tea_hook, _ = build_teacher(CKPT_SCENE, DEVICE)
#     student, stu_hook, d_s = build_student(DEVICE)
#     disc = TemporalDiscriminator(in_dim=d_s).to(DEVICE)
#
#     lE = build_subset_loader_event(EVENT_KEEP_RATIO, BATCH_SIZE, T, DEVICE)
#     lS = build_subset_loader_scene(SPLIT_PATH_SCENE, SCENE_KEEP_RATIO, BATCH_SIZE, T)
#
#     optM = torch.optim.Adam(student.parameters(), lr=LR)
#     optD = torch.optim.Adam(disc.parameters(), lr=LR)
#     schM = torch.optim.lr_scheduler.LambdaLR(
#         optM, warmup_decay_cosine(len(lE), len(lE)*EPOCHS)
#     )
#     schD = torch.optim.lr_scheduler.LambdaLR(
#         optD, warmup_decay_cosine(len(lE), len(lE)*EPOCHS)
#     )
#
#     for ep in range(EPOCHS):
#         train_epoch_schemeB(lE, lS, student, stu_hook,
#                             teacher, tea_hook, disc, optM, optD, schM, schD)
#         ckpt_path = os.path.join(OUT_DIR, f"student_ep{ep:03d}.pth")
#         torch.save({"model": student.state_dict(), "epoch": ep}, ckpt_path)
#         print(f"✅ Saved checkpoint to {ckpt_path}")
#     scene_head = MlpHead(in_dim=2176, hid_dim=512).to(f"cuda:{gpu}")
#     for n, p in student.named_parameters():
#         p.requires_grad = False
#     student.detector = scene_head
#     for p in student.detector.parameters():
#         p.requires_grad = True
#
#     criterion = nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.Adam(student.detector.parameters(), lr=1e-4,
#                                  betas=(0.9, 0.98), weight_decay=1e-4)
#     total_steps = len() * epochs
#     scheduler = torch.optim.lr_scheduler.LambdaLR(
#         optimizer, lr_lambda=warmup_decay_cosine(len(scene_train_loader), total_steps))
#
#     for ep in range(1, epochs + 1):
#         train_one_epoch(scene_train_loader, model, optimizer, scheduler, criterion, gpu=gpu, epoch=ep)
#         scene_metrics = evaluate_scene(scene_test_loader, model, gpu=gpu)
#         print(f"\n[Scene Eval after Epoch {ep}] Metrics: {scene_metrics}")
#
#     print("\n✅ Training + Evaluation Pipeline Completed.")
# # def main():
# #     teacher, tea_hook, _ = build_teacher(CKPT_SCENE, DEVICE)
# #     student, stu_hook, d_s = build_student(DEVICE)
# #     disc = TemporalDiscriminator(in_dim=d_s).to(DEVICE)
# #
# #     loader_event = build_subset_loader_event(EVENT_KEEP_RATIO, BATCH_SIZE, T, DEVICE)
# #     loader_scene = build_subset_loader_scene(SPLIT_PATH_SCENE, SCENE_KEEP_RATIO, BATCH_SIZE, T)
# #
# #     opt_student = torch.optim.Adam(student.parameters(), lr=LR)
# #     opt_disc = torch.optim.Adam(disc.parameters(), lr=LR)
# #
# #     sch_student = torch.optim.lr_scheduler.LambdaLR(
# #         opt_student, warmup_decay_cosine(len(loader_scene), len(loader_scene)*EPOCHS)
# #     )
# #     sch_disc = torch.optim.lr_scheduler.LambdaLR(
# #         opt_disc, warmup_decay_cosine(len(loader_scene), len(loader_scene)*EPOCHS)
# #     )
# #
# #     for ep in range(EPOCHS):
# #         train_epoch_schemeA_mixed(loader_scene, loader_event,
# #                                   student, stu_hook,
# #                                   teacher, tea_hook,
# #                                   disc,
# #                                   opt_student, opt_disc,
# #                                   sch_student, sch_disc)
# #         ckpt_path = os.path.join(OUT_DIR, f"student_ep{ep:03d}.pth")
# #         torch.save({"model": student.state_dict(), "epoch": ep}, ckpt_path)
# #         print(f"✅ Saved checkpoint to {ckpt_path}")
# #
# # import os
# # import argparse
# # from collections import defaultdict
# # import torch
# # from torch import nn
# # import numpy as np
# # from tqdm import tqdm
# # from torch.utils.data import DataLoader
# #
# # # ===== 模型与工具依赖 =====
# # from model.backbone.RelationNet import LocalUncertaintyAwareGraphAttentionLite
# # from model.detector.scene_detector import MlpHead
# # from dataset.kinetics.dataset import build as build_kinetics, collate_fn
# # from dataset.movienet.load_movienet_server import load_data
# # from tool.warmup_lr import warmup_decay_cosine
# # from tool.metric.scene_metric import metric as scene_metric
# # # ===============================================================
# # # Scene Head 训练 & 评估
# # # ===============================================================
# # def train_one_epoch(scene_loader, model, optimizer, scheduler, criterion, gpu=0, epoch=1):
# #     """scene_head 单次训练(一轮)"""
# #     model.train().cuda(gpu)
# #     epoch_loss = 0.0
# #     for batch_idx, batch in enumerate(tqdm(scene_loader, desc=f"🎯 Train Scene Epoch {epoch}")):
# #         names, img_ctx, plc_ctx, label, pos, ids, ind = batch = batch
# #         img_ctx, plc_ctx, label = img_ctx.cuda(gpu), plc_ctx.cuda(gpu), label.float().cuda(gpu)
# #
# #         optimizer.zero_grad()
# #         x = 0.7 * img_ctx + 0.3 * plc_ctx
# #         logtics, pred = model(x)
# #         loss = criterion(logtics.squeeze(), label)
# #         loss.backward()
# #         optimizer.step()
# #         scheduler.step()
# #         epoch_loss += loss.item()
# #
# #     avg_loss = epoch_loss / len(scene_loader)
# #     lr = scheduler.get_last_lr()[0]
# #     print(f"[Scene Train] Epoch {epoch}: Loss={avg_loss:.4f}, LR={lr:.6f}")
# #     return avg_loss
# #
# #
# # @torch.no_grad()
# # def evaluate_scene(scene_loader, model, gpu=0):
# #     """Scene Head 推理并计算 metric"""
# #     model.eval().cuda(gpu)
# #     predlist, labelist, pathlist = [], [], []
# #     for batch in tqdm(scene_loader, desc="Scene Eval"):
# #         names, img_ctx, plc_ctx, label, pos, ids, ind = batch = batch
# #         img_ctx, plc_ctx = img_ctx.cuda(gpu), plc_ctx.cuda(gpu)
# #         x = 0.7 * img_ctx + 0.3 * plc_ctx
# #         _, pred = model(x)
# #         predlist.append(pred.cpu().numpy())
# #         labelist.append(label.numpy())
# #         pathlist.append(names)
# #     met, moviePL = scene_metric(pathlist, predlist, labelist)
# #     return met
# #
# # # ===============================================================
# # # 主程序
# # # ===============================================================
# # def main():
# #     gpu = 5
# #     ckpt_path = "/home/tianxiaoxuan/data/mamba/checkpoint_event_BNA/checkpoint_event_BNA/ckpt_ep0.pth"
# #     feature_path = "/data/shared_dataset/Kinetics/features"
# #     anno_path = "/data/shared_dataset/Kinetics/data"
# #     img_path = "/data/shared_dataset/shared_dataset/MovieDatasets/MovieNet/ImageNet_shot.pkl"
# #     plc_path = "/data/shared_dataset/shared_dataset/MovieDatasets/MovieNet/Places_shot.pkl"
# #     label_path = "/home/tianxiaoxuan/data/mamba/data/label_endShot.pkl"
# #     split_path = "/home/tianxiaoxuan/data/mamba/data/split318.json"
# #     mamba_path = "/home/tianxiaoxuan/data/mamba/data/ImageNet_shot.pkl"
# #     window_size = 21
# #     epochs = 10
# #
# #     # ==== 加载主干 ====
# #     print(f"\n🚀 Loading checkpoint: {ckpt_path}")
# #     model = LocalUncertaintyAwareGraphAttentionLite(
# #         dim_in=2048, num_heads=8, window_size=5,
# #         sim_temperature=0.07, neighbor_temp=0.5,
# #         uncertainty_mode="variance", use_relative_pos_bias=True,
# #         norm="ln", size=window_size,
# #     ).to(f"cuda:{gpu}")
# #     ckpt = torch.load(ckpt_path, map_location="cpu")
# #     if isinstance(ckpt, dict) and "model" in ckpt:
# #         state_dict = ckpt["model"]
# #     else:
# #         state_dict = ckpt  # 纯 state_dict
# #
# #     msg = model.load_state_dict(state_dict, strict=False)
# #     print("checkpoint loaded:", msg)
# #     model.load_state_dict(ckpt, strict=False)
# #     print("✅ Loaded backbone and pre-trained event detector from checkpoint.")
# #
# #     # ==== 构建数据 ====
# #     scene_train_loader = load_data(label_path, img_path, plc_path, split_path,
# #                                    batch=128, seg_sz=window_size, mode1="train", mode2=None)
# #     scene_test_loader = load_data(label_path, img_path, plc_path, split_path,
# #                                   batch=128, seg_sz=window_size, mode1="test", mode2=None)
# #
# #     # =======================================================
# #     # 2️⃣ Scene Head fine-tune & evaluate each epoch
# #     # =======================================================
# #     print("\n🎞 Fine-tuning Scene Head ...")
# #     scene_head = MlpHead(in_dim=2176, hid_dim=512).to(f"cuda:{gpu}")
# #     for n, p in model.named_parameters():
# #         p.requires_grad = False
# #     model.detector = scene_head
# #     for p in model.detector.parameters():
# #         p.requires_grad = True
# #
# #     criterion = nn.BCEWithLogitsLoss()
# #     optimizer = torch.optim.Adam(model.detector.parameters(), lr=1e-4,
# #                                  betas=(0.9, 0.98), weight_decay=1e-4)
# #     total_steps = len(scene_train_loader) * epochs
# #     scheduler = torch.optim.lr_scheduler.LambdaLR(
# #         optimizer, lr_lambda=warmup_decay_cosine(len(scene_train_loader), total_steps))
# #
# #     for ep in range(1, epochs + 1):
# #         train_one_epoch(scene_train_loader, model, optimizer, scheduler, criterion, gpu=gpu, epoch=ep)
# #         scene_metrics = evaluate_scene(scene_test_loader, model, gpu=gpu)
# #         print(f"\n[Scene Eval after Epoch {ep}] Metrics: {scene_metrics}")
# #
# #     print("\n✅ Training + Evaluation Pipeline Completed.")
# #
# #     # =======================================================
# #     # 2️⃣ Scene Head fine-tune & evaluate each epoch
# #     # =======================================================
# #     print("\n🎞 Fine-tuning Scene Head ...")
# #     scene_head = MlpHead(in_dim=2176, hid_dim=512).to(f"cuda:{gpu}")
# #     for n, p in model.named_parameters():
# #         p.requires_grad = False
# #     model.detector = scene_head
# #     for p in model.detector.parameters():
# #         p.requires_grad = True
# #
# #     criterion = nn.BCEWithLogitsLoss()
# #     optimizer = torch.optim.Adam(model.detector.parameters(), lr=1e-4,
# #                                  betas=(0.9, 0.98), weight_decay=1e-4)
# #     total_steps = len(scene_train_loader) * epochs
# #     scheduler = torch.optim.lr_scheduler.LambdaLR(
# #         optimizer, lr_lambda=warmup_decay_cosine(len(scene_train_loader), total_steps))
# #
# #     for ep in range(1, epochs + 1):
# #         train_one_epoch(scene_train_loader, model, optimizer, scheduler, criterion, gpu=gpu, epoch=ep)
# #         scene_metrics = evaluate_scene(scene_test_loader, model, gpu=gpu)
# #         print(f"\n[Scene Eval after Epoch {ep}] Metrics: {scene_metrics}")
# #
# #     print("\n✅ Training + Evaluation Pipeline Completed.")
# #
# #
# # if __name__ == "__main__":
# #     main()
# #
