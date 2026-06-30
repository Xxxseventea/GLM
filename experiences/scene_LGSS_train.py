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
from typing import Dict, Any
# from model.backbone.RelationNet import LocalUncertaintyAwareGraphAttentionLite
# from model.detector.scene_detector import MlpHead
import json
# from model.PC_new import PCModel
from model.lgss_event import LGSSEventDet 

from datetime import datetime
# 配置
EPOCHS = 30
BATCH_SIZE = 32
T = 20
GPU = 0

# 路径
movienet_path = "data/MovieNet/"
IMG_PATH = movienet_path + 'ImageNet_shot.pkl'
PLC_PATH = movienet_path + 'Places_shot.pkl'
LABEL_PATH = movienet_path + 'label_endShot.pkl'
SPLIT_PATH = movienet_path + 'split318.json'
MAMBA_PATH = movienet_path + 'ImageNet_shot.pkl'

CKPT_DIR = '<weight/checkpoint.pt>'
SIDE_DIR = '<side_outputs>'
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(SIDE_DIR, exist_ok=True)
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


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
            logtis, _ = model(img_ctx)
            pred = torch.sigmoid(logtis).squeeze(-1)

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

def train_epoch(trainload, model, opti, lr_sh, epoch, gpu=GPU):
    model.train()
    progress = tqdm(trainload)
    num_updates = epoch * len(trainload)
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
        logtis, _ = model(img_ctx)
        logtis = logtis.squeeze(-1)
        # 单一 BCE
        loss= F.binary_cross_entropy_with_logits(logtis, label.float())

        loss.backward()
        opti.step()
        num_updates += 1
        progress.set_postfix(loss=f'{loss.item():.6f}')

    if lr_sh is not None:
        lr_sh.step()
    return 1


class Scheduler:
    """ Parameter Scheduler Base Class
    A scheduler base class that can be used to schedule any optimizer parameter groups.
    Unlike the builtin PyTorch schedulers, this is intended to be consistently called
    * At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
    * At the END of each optimizer update, after incrementing the update count, to calculate next update's value
    The schedulers built on this should try to remain as stateless as possible (for simplicity).
    This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
    and -1 values for special behaviour. All epoch and update counts must be tracked in the training
    code and explicitly passed in to the schedulers on the corresponding step or step_update call.
    Based on ideas from:
     * https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler
     * https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize: bool = True) -> None:
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.metric = None  # any point to having this for all?
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)

    def state_dict(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def get_epoch_values(self, epoch: int):
        return None

    def get_update_values(self, num_updates: int):
        return None

    def step(self, epoch: int, metric: float = None) -> None:
        self.metric = metric
        values = self.get_epoch_values(epoch)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates: int, metric: float = None):
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
            if apply_noise:
                g = torch.Generator()
                g.manual_seed(self.noise_seed + t)
                if self.noise_type == 'normal':
                    while True:
                        # resample if noise out of percent limit, brute force but shouldn't spin much
                        noise = torch.randn(1, generator=g).item()
                        if abs(noise) < self.noise_pct:
                            break
                else:
                    noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
                lrs = [v + v * noise for v in lrs]
        return lrs
    

class StepLRScheduler(Scheduler):
    """
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 decay_t: float,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.decay_t = decay_t
        self.decay_rate = decay_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [v * (self.decay_rate ** (t // self.decay_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
        

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


    # 实例化 UAG：两层、窗口半径 6（窗口长度 13）、每步保留 8 个邻居
    model = LGSSEventDet(shot_num=20, place_feat_dim=2048, sim_channel=512)
    model.to(f"cuda:{GPU}")
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-2, betas=(0.9, 0.98), weight_decay=5e-4,
    )
    lr_sh = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,milestones=[15]
    )

    # 4) 训练/验证循环
    epochs = 30
    bce_pos_weight = 5  # 如需正样本加权，设 torch.tensor([w]).to(args.device)


    train_dataLoader = load_data(LABEL_PATH, IMG_PATH, PLC_PATH, SPLIT_PATH,
                         BATCH_SIZE, seg_sz=T, mode1='train', mode2=None)
    test_dataLoader = load_data(LABEL_PATH, IMG_PATH, PLC_PATH, SPLIT_PATH,
                        BATCH_SIZE, seg_sz=T, mode1='test', mode2=None)

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

    for i in range(epochs):
        train_epoch(trainload=train_dataLoader, model=model, opti=optimizer, lr_sh=lr_sh, gpu=GPU, epoch=i)
        ckpt_path = os.path.join(CKPT_DIR, f'epoch_{i:03d}.pt')
        side_path = os.path.join(SIDE_DIR, f'eval_{i:03d}.npz')
        met, moviePL = test_epoch(testload=test_dataLoader, model=model, gpu=GPU, side_save_path=side_path)
        # lr_sh.step(i+1, met["F1"])
        print(f"[Epoch {i}] metric = {met}")
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

        ckpt = {
            "epoch": i,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
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
        scene_eval_record = {
            "stage": "phase1_scene_training",
            "epoch": EPOCHS,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metric": met,
            "moviePL": moviePL,
        }

        save_json(
            scene_eval_record,
            os.path.join(CKPT_DIR, f"scene_eval_final_ep{i}.json")
        )


if __name__ == '__main__':
    main()

