import numpy as np
import pickle as pkl
import random
import torch
import os


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, ft_img:dict, ft_plc:dict,trainSet:list, seg_sz=20, mode1='train', mode2='pretrain'):
        """
        :param ft_img: dict, imagenet or modality-A features, {movie_id:{shot_id:feature}}
        :param ft_plc: dict, cut or Modality-B features, same as above
        :param trainSet: movie ids, [tt0xxxx,,...]
        :param seg_sz: scale of time window, noted that seg_sz is odd number
        """
        self.trainSet = trainSet
        self.imgft = ft_img
        self.plcft = ft_plc
        self.seg_sz = seg_sz
        self.win = seg_sz//2
        self.samplelist = self._gen_datalist(mode2)
        self.mode1 = mode1
        self.mode2 = mode2
        if mode1 == 'train':
            random.shuffle(self.samplelist)

    def __getitem__(self, ind):
        return self.samplist[ind]

    def __len__(self):
        return len(self.samplelist)

    def _read_pkl(self, path):
        with open(path, 'rb') as f:
            data = pkl.load(f)
        return data

    def load_clip(self, name, ids):
        """
        :param name: movie_id
        :param ids:
        :return:
            vec:torch.Tensor, (seg_sz, dim(modality-A)+dim(modality-B))
        """
        img_vec = []
        plc_vec = []
        mamba_vec = []
        for shid in ids:
            img_vec.append(torch.from_numpy(self.imgft[name][f'{shid:04d}'])[None])
            plc_vec.append(torch.from_numpy(self.plcft[name][f'{shid:04d}'])[None])
        img_vec = torch.concat(img_vec, dim=0)
        plc_vec = torch.concat(plc_vec, dim=0)
        return img_vec, plc_vec

    def _gen_datalist(self, mode):
        """
        :return:
            datalist: list, [movie_name, center]
        """
        # if mode == 'pretrain':
        #     datalist = []
        #     img_k = set(self.imgft)
        #     plc_k = set(self.plcft)
        #     mamba_k = set(self.mambaft)
        #     f_k = img_k.intersection(plc_k).intersection(mamba_k)
        #     half = self.win
        #     for movie in self.trainSet:
        #         if movie in f_k:
        #             n_shot = len(self.imgft[movie].keys())
        #             for s in range(0, n_shot - self.seg_sz, 1):
        #                 datalist.append({
        #                     'vid': movie,
        #                     'begin_shid': s,
        #                     'all_shids':np.clip(np.arange(s, s + self.seg_sz), 0, n_shot - 1)
        #                 })
        # else:
        datalist = []
        img_k = set(self.imgft)
        plc_k = set(self.plcft)
        f_k = img_k.intersection(plc_k)
        half = self.win
        for movie in self.trainSet:
            if movie in f_k:
                n_shot = len(self.imgft[movie].keys())
                for i in range(half, n_shot - half):
                    ctx_id = np.arange(i - half, i + half + 1)
                    ctx_id = np.clip(ctx_id, 0, n_shot - 1)
                    datalist.append([movie, i, ctx_id])
        print('Finish generating pretrain list!')


        return datalist
