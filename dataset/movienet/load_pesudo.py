import numpy as np
import pickle as pkl

import pandas as pd
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
import os
import json as js
import torch
from dataset.BaseDataset import BaseDataset

def read_pkl(path):
    with open(path , 'rb') as f:
        data = pkl.load(f)
    return data

def read_pkl2(path):
    data = {}
    with open(path, 'rb') as f:
        while True:
            try:
                data1 = pkl.load(f)
                data.update(data1)
            except EOFError:
                break
    return data

class MovieNetDataset(BaseDataset):
    def __init__(self, pesudo_bound, label_dict, ft_img:dict, ft_plc:dict,  splitSet:list, seg_sz=21, mode1='train', mode2='pretrain'):
        super().__init__(ft_img, ft_plc, splitSet, seg_sz, mode1, mode2)
        self.pesudo_bound = pesudo_bound
        self.label_dict = label_dict
    def __getitem__(self, ind):
        samplelist = self.samplelist[ind]
        begin_shid = samplelist['begin_shid']
        vid = samplelist['vid']
        all_shids = samplelist['all_shids']

        centre_shid = begin_shid + self.seg_sz // 2

        ## 特征
        img_ctx,plc_ctx = self.load_clip(vid, all_shids)
        img_ctx = img_ctx.to(torch.float)
        plc_ctx = plc_ctx.to(torch.float)

        ## 正负标签idx

        pseudo_bound = self.pesudo_bound[vid][begin_shid]

        pos_idx = pseudo_bound
        neg_idx = np.random.randint(self.seg_sz-1)
        if neg_idx == pseudo_bound: neg_idx += 1


        ## 标签
        return img_ctx, plc_ctx, pos_idx, neg_idx


def load_data(pesudo_bound_path, label_dict_path, anno_path,  modalA_path, modalB_path, split_path, batch, mode1='train', mode2='pretrain', seg_sz=21):
    with open(split_path, 'r') as f:
        data = js.load(f)
    with open(anno_path, 'r') as f:
        anno = js.load(f)
    bad_vids = ['tt0095016', 'tt0117951', 'tt0120755'] + ['tt0258000', 'tt0120263'] + ['tt3465916']
    if mode1 == 'train':
        splitSet = [vid for vid in data['train'] if vid not in anno['all'] and vid not in bad_vids]
    else:
        splitSet = [vid for vid in data['test'] if vid not in anno['all'] and vid not in bad_vids]
    modalA_feat = read_pkl2(modalA_path)
    modalB_feat = read_pkl2(modalB_path)
    pseudo_labels = read_pkl(pesudo_bound_path)
    label_dict = read_pkl(label_dict_path)
    Dataset = MovieNetDataset(pseudo_labels, label_dict, modalA_feat, modalB_feat,  splitSet, seg_sz, mode1 ='train', mode2 = 'pretrain')
    if mode1 == 'train':
        dataLoader = torch.utils.data.DataLoader(Dataset, batch_size=batch,
                                                 shuffle=True, drop_last=True ,num_workers=0)
    else:
        dataLoader = torch.utils.data.DataLoader(Dataset, batch_size=batch,
                                                 shuffle=False, drop_last=False, num_workers=0)

    return dataLoader



