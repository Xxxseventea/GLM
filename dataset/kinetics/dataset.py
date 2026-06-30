
"""
Modified from RTD-Net (https://github.com/MCG-NJU/RTD-Action/blob/main/datasets/thumos14.py)

KineticsGEBD dataset which returns sample list with annotation for training and validation.
Now supports including negative samples (windows with no GT).
"""

from pathlib import Path

import argparse
import torch
import torch.utils.data
import json
import pandas as pd
import os
import numpy as np
import copy
import pickle
from typing import List, Dict, Any


def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def normalize_gts_to_unit_interval(gts_abs: List[float], win_locations: np.ndarray) -> List[float]:
    """
    将窗口内的绝对帧 GT 列表映射为 [0,1] 相对坐标。
    依据真实 locations 跨度线性归一化，确保与测试时插值一致。
    输入:
      - gts_abs: 绝对帧编号（float/int）
      - win_locations: 窗口内 locations (L,) 为绝对帧编号
    返回:
      - y_list: List[float], 每个 y ∈ [0,1]
    """
    win_left = float(win_locations.min())
    win_right = float(win_locations.max())
    span = max(win_right - win_left, 1e-6)
    y_list = [(float(g) - win_left) / span for g in gts_abs]
    # 由于浮点误差，轻微裁剪到 [0,1]
    y_list = [min(max(y, 0.0), 1.0) for y in y_list]
    return y_list


class VideoRecord:
    def __init__(self, vid, num_frames, locations, gt_abs_frames, coherence_scores, fps, args):
        """
        locations: np.ndarray shape (L,), 绝对帧编号
        gt_abs_frames: List[int or float], 窗口内的绝对帧 GT 中心（若无则空列表）
        """
        self.id = vid                               # 字符串视频 ID
        self.locations = locations.astype(np.float32)
        self.base = float(self.locations[0])        # 用于兼容下游接口
        self.window_size = args.window_size
        self.interval = getattr(args, "interval", 1)
        # 相对 base 的 locations（仅用于兼容 collate_fn 的原输出）
        self.rel_locations = [float(location - self.base) for location in self.locations]
        self.num_frames = num_frames

        # 原始 GT（绝对帧）
        self.gt_abs_frames = list(gt_abs_frames)    # [g1, g2, ...]（可能为空）

        # 与推理一致：用 locations 的真实跨度将 GT 归一化到 [0,1]
        self.gt_frames_unit = normalize_gts_to_unit_interval(self.gt_abs_frames, self.locations)

        self.fps = fps

        # coherence_scores 归一化（与原代码一致，防止除 0）
        cs = np.asarray(coherence_scores, dtype=np.float32)
        if cs.size > 0:
            rng = float(cs.max() - cs.min()) if float(cs.max() - cs.min()) > 0 else 1.0
            self.coherence_scores = (cs - cs.min()) / rng
        else:
            self.coherence_scores = cs


class KineticsGEBD(torch.utils.data.Dataset):
    def __init__(self, feature_folder, score_path, anno_root_path, mode, args):
        """
        新增行为：
        - 训练集默认保留负样本（keep_neg_in_train=True），可通过 args.keep_neg_in_train 控制。
        - 验证/测试集保留所有窗口（与原行为一致）。
        """
        self.window_size = args.window_size
        self.feature_folder = feature_folder
        self.mode = mode
        self.keep_neg_in_train = getattr(args, "keep_neg_in_train", True)  # 默认保留负样本

        # 读取标注与帧序列
        anno_path = os.path.join(anno_root_path, f'k400_mr345_{mode}_min_change_duration0.3.pkl')
        with open(anno_path, 'rb') as f:
            dict_ann = pickle.load(f, encoding='lartin1')  # 原代码中的编码

        video_pool = list(dict_ann.keys())
        video_pool.sort()

        self.sample_list: List[VideoRecord] = []

        # coherence 分数与采样帧
        with open(os.path.join(score_path, f'{mode}_score_sequence.pkl'), 'rb') as f:
            scores = pickle.load(f, encoding='lartin1')

        for vid in video_pool:
            vdict = dict_ann[vid]
            num_frames = vdict['num_frames']
            fps = vdict['fps']

            f1_consis = vdict['f1_consis']
            highest = int(np.argmax(f1_consis))
            annotations = vdict['substages_myframeidx'][highest]  # 绝对帧 GT 中心列表
            # labels = [1 for _ in annotations]  # 原逻辑中 label 恒为 1，这里不再需要

            coherence_scores_per_vid = np.asarray(scores[vid]['scores'], dtype=np.float32)
            frames = np.asarray(scores[vid]['frame_idx'], dtype=np.float32)  # 绝对帧编号列表（等长于 scores）
            num_sampled = len(frames)

            # 小于等于窗口长度：padding
            if num_sampled <= self.window_size:
                locations = np.zeros((self.window_size,), dtype=np.float32)
                locations[:num_sampled] = frames
                coherence_scores = np.zeros((self.window_size,), dtype=np.float32)
                coherence_scores[:num_sampled] = coherence_scores_per_vid

                # 该窗口内的 GT（绝对帧）筛选
                left, right = float(locations.min()), float(locations.max())
                gt_abs = [float(a) for a in annotations if (a >= left and a <= right)]

                # 训练模式下是否保留负样本
                if (self.mode != 'train') or (self.keep_neg_in_train or len(gt_abs) > 0):
                    self.sample_list.append(VideoRecord(
                        vid, num_frames, locations, gt_abs, coherence_scores, fps, args
                    ))
            else:
                # 滑窗
                overlap_ratio = 1
                stride = max(self.window_size // overlap_ratio, 1)
                ws_starts = [i * stride for i in range((num_sampled // self.window_size - 1) * overlap_ratio + 1)]
                ws_starts.append(num_sampled - self.window_size)

                for ws in ws_starts:
                    locations = frames[ws:ws + self.window_size]
                    coherence_scores = coherence_scores_per_vid[ws:ws + self.window_size]

                    left, right = float(locations.min()), float(locations.max())
                    # 窗口内 GT（绝对帧）
                    gt_abs = [float(a) for a in annotations if (a >= left and a <= right)]

                    # 训练：保留负样本可控；验证：全保留
                    if (self.mode != 'train') or (self.keep_neg_in_train or len(gt_abs) > 0):
                        self.sample_list.append(VideoRecord(
                            vid, num_frames, locations, gt_abs, coherence_scores, fps, args
                        ))

    def get_data(self, video: VideoRecord):
        """
        返回:
          - locations: [L, 1], 相对 base 的偏移（double），与原接口保持一致
          - features: [L, C]，索引自整段视频特征
          - targets: dict{'labels': LongTensor(#gt), 'boundaries': Tensor(#gt in [0,1]), 'video_id': str}
          - num_frames: int
          - base: float(窗口首帧的绝对帧)
          - coherence_scores: [L]
        """

        vid = video.id
        num_frames = video.num_frames
        base = video.base

        # 绝对帧 locations
        abs_locations = torch.tensor(video.locations, dtype=torch.long)  # 用于映射到特征索引
        # 读取视频特征
        vid_feature_path = os.path.join(self.feature_folder, vid)
        vid_feature = torch.load(vid_feature_path, map_location="cpu", weights_only=True)
        if isinstance(vid_feature, dict) and 'feat' in vid_feature:
            vid_feature = vid_feature['feat']
        vid_feature = torch.as_tensor(vid_feature)

        # 标准化到 [T, C]
        if vid_feature.ndim == 3:
            if vid_feature.shape[0] == 1:
                vid_feature = vid_feature[0]        # [T, C]
            elif vid_feature.shape[-1] == 1:
                vid_feature = vid_feature[..., 0]   # [T, C]
            else:
                vid_feature = vid_feature.squeeze()
        if vid_feature.ndim != 2:
            raise ValueError(f"Feature file {vid} has shape {tuple(vid_feature.shape)}; expected [T, C].")

        T_steps, ft_dim = vid_feature.shape

        # 将绝对帧映射到特征步索引（步=帧//step_frames）
        step_frames = 3  # 若特征提取是每 3 帧一个步长
        ft_idxes = torch.div(abs_locations, step_frames, rounding_mode='floor').clamp_(0, T_steps - 1).long()
        features = vid_feature.index_select(dim=0, index=ft_idxes)  # [L, C]

        # 你可加入一致性断言（按需修改 ft_dim）
        # assert features.shape[0] == len(video.locations), f"{features.shape} vs {len(video.locations)}"

        # 相对 base 的 locations（double）以兼容原 collate_fn
        locations = torch.tensor(video.rel_locations, dtype=torch.double)  # [L]
        coherence_scores = torch.tensor(video.coherence_scores, dtype=torch.float32)  # [L]

        # 构造 targets
        # gt_frames_unit: List[float] ∈ [0,1]，与测试时插值一致的相对坐标
        targets = {'labels': [], 'boundaries': [], 'video_id': vid}
        # labels 恒为 1（GEBD 的边界事件），负样本时列表为空
        for y in video.gt_frames_unit:
            targets['labels'].append(1)
            targets['boundaries'].append(float(y))
        targets['labels'] = torch.LongTensor(targets['labels']) if len(targets['labels']) > 0 else torch.LongTensor([])
        targets['boundaries'] = torch.tensor(targets['boundaries'], dtype=torch.float32) if len(targets['boundaries']) > 0 else torch.tensor([], dtype=torch.float32)

        return locations, features, targets, num_frames, base, coherence_scores

    def __getitem__(self, idx):
        return self.get_data(self.sample_list[idx])

    def __len__(self):
        return len(self.sample_list)


def collate_fn(batch):
    """
    将 batch 样本堆叠为批处理张量，与原接口保持一致。
    """
    target_list, num_frames_list, base_list = [[] for _ in range(3)]
    batch_size = len(batch)
    ft_dim = batch[0][1].shape[-1]
    max_props_num = batch[0][0].shape[0]

    features = torch.zeros(batch_size, max_props_num, ft_dim, dtype=batch[0][1].dtype)
    locations = torch.zeros(batch_size, max_props_num, 1, dtype=torch.double)
    coherence_scores = torch.zeros(batch_size, max_props_num, dtype=torch.float32)

    for i, sample in enumerate(batch):
        loc_i, feat_i, tgt_i, nf_i, base_i, coh_i = sample
        # locations: [L] → [L,1]
        locations[i, :max_props_num, :] = loc_i.reshape((-1, 1))
        features[i, :max_props_num, :] = feat_i
        target_list.append(tgt_i)
        num_frames_list.append(nf_i)
        base_list.append(base_i)
        coherence_scores[i, :max_props_num] = coh_i

    num_frames = torch.from_numpy(np.array(num_frames_list))
    base = torch.from_numpy(np.array(base_list, dtype=np.float64))

    return locations, features, target_list, num_frames, base, coherence_scores


def build(split, args):
    feature_folder = Path(args.feature_path)
    score_path = Path(args.score_path)
    anno_file = Path(args.annotation_path)

    dataset = KineticsGEBD(feature_folder, score_path, anno_file, split, args)
    if(split == 'train'):
        print(len(dataset))
    return dataset
