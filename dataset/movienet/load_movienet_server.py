import random

import numpy as np
import pickle as pkl

import pandas as pd
from torch.utils.data import SubsetRandomSampler
from tqdm import tqdm
import os
import json as js
import torch
from dataset.movienet.BaseDataset import BaseDataset

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
    def __init__(self, labels:dict, ft_img:dict, ft_plc:dict, splitSet:list, seg_sz=20, mode1='train', mode2=None):
        super().__init__(ft_img, ft_plc, splitSet, seg_sz, mode1, mode2)
        self.mvlabels = labels

    def __getitem__(self, ind):
        name, cid, ids = self.samplelist[ind]
        img_ctx,plc_ctx = self.load_clip(name, ids)
        img_ctx = img_ctx.to(torch.float)
        plc_ctx = plc_ctx.to(torch.float)
        label = self.mvlabels[name][f'{cid:04d}']

        if label == -1:
            label = 1
        label = torch.from_numpy(np.array(label))
        label = label.to(torch.float)
        pos = 10


        return name+'shot'+str(cid), img_ctx,plc_ctx, label, pos, ids,ind


def load_data(label_path, modalA_path, modalB_path, split_path, batch, mode1='train', mode2=None, seg_sz=20):
    with open(split_path, 'r') as f:
        data = js.load(f)
        if mode1 == 'train':
            splitSet = data['train']  + data['val']
        else:
            splitSet = data['test']

    modalA_feat = read_pkl2(modalA_path)
    modalB_feat = read_pkl2(modalB_path)
    labels = read_pkl(label_path)
    Dataset = MovieNetDataset(labels, modalA_feat, modalB_feat, splitSet, seg_sz, mode1, mode2)
    if mode1 == 'train':
        n = len(Dataset)
        print(n)
        dataLoader = torch.utils.data.DataLoader(Dataset, batch_size=batch,
                                                 shuffle=True, drop_last=True ,num_workers=0)
    else:
        dataLoader = torch.utils.data.DataLoader(Dataset, batch_size=batch,
                                                 shuffle=False, drop_last=False, num_workers=0
                                                 )

    return dataLoader


# if __name__=='__main__':
#     img_path = '/data/shared_dataset/MovieDatasets/MovieNet/ImageNet_shot.pkl'
#     cut_path = '/data/tianxiaoxuan/data/cut_features.pkl'
#     label_path = '/data/tianxiaoxuan/data/label_endShot.pkl'
#     split_path = '/data/tianxiaoxuan/data/split318.json'
#
#     dataLoader = load_data(label_path, img_path, cut_path, split_path, 128, seg_sz=13, mode='test')
#
#     for data in dataLoader:
#         names, ctx, label = data
#         print(ctx.shape)



