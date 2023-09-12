# -*- coding:utf-8 -*-
# Description:
# Author: WZJ
# Date:   2021/12/9 14:09
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import os

__all__ = ['MemoryBlock']


class MemoryBlock(nn.Module):
    def __init__(self, memory_init,moving_rate=0.999):
        super().__init__()
        # self.kdim = kdim  # slot
        # self.hdim = hdim  # dimension
        self.moving_rate = moving_rate
        # feat_dir='data_all/text_features'
        # features=[]
        # for file in os.listdir(feat_dir):
        #     feat_name=os.path.join(feat_dir,file)
        #     features.append(torch.load(feat_name))
        # self.memory=features
        # self.memory=torch.load('/mnt/workspace/workgroup/zhibing/Anyface/mapper/data_all/memory_features.pt')
        self.memory=torch.load(memory_init)
    def update(self, value, score, memory, m=None):
        '''
            x: (1, 256)
            e: (64, 256)
            score: (1, 64)
        '''
        # if m is None:
        #     m = getattr(self, memory).weight.data
        value = value.detach()
        embed_ind = torch.max(score, dim=1)[1]  # (n, )
        embed_onehot = F.one_hot(embed_ind, m.size()[0]).type(value.dtype)  # (n, k)
        embed_onehot_sum = embed_onehot.sum(0)
        embed_sum = value.transpose(0, 1) @ embed_onehot  # (c, k)
        embed_mean = embed_sum / (embed_onehot_sum + 1e-6)
        new_data = m * self.moving_rate + embed_mean.t() * (1 - self.moving_rate)

        # if self.training:
        #     self.memory.weight.data = new_data
        # getattr(self, memory).weight.data = new_data
        self.memory[memory]=new_data

        return new_data

    def forward(self, key, value, update=True):
        '''
            x: style code, (14, 256)
        '''
        out = torch.zeros_like(key, dtype=key.dtype, device=key.device)
        for i in range(key.size()[1]):
            # memory = 'memory_{:d}'.format(i)
            # m = getattr(self, memory).weight.data
            m=self.memory[i]
            xn = F.normalize(key[:, i, :], dim=1)
            mn = F.normalize(m, dim=1)
            score = torch.matmul(xn, mn.t())
            if update:
                m = self.update(value[:, i, :], score, i, m)
                mn = F.normalize(m, dim=1)
                score = torch.matmul(xn, mn.t())
            embed_ind = torch.max(score, dim=1)[1]
            tough_label = F.one_hot(embed_ind, m.size()[0]).type(value.dtype)
            # soft_label = F.softmax(score, dim=1)
            out[:, i, :] = value[:,0,:].squeeze()+torch.matmul(tough_label, m)
        # out = torch.cat([key, out], dim=-1)
        return {"mask":key, "unmask":value, "memory":out, 'memory_all': self.memory}
