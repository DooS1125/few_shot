import torch
import numpy as np
from Data import *
import pandas as pd


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind) # argwhere을 통해 해당 label을 가지고 있는 index값을 각각 담음, 즉 m_ind는 각 클래스별 데이터의 인덱스 값을 가지고 있음

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls] #전체 클래스에서 랜덤하게 n_cls만큼 클래스 추출(way) 개념)
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per] # 랜덤하게 뽑힌 클래스에 해당하는 데이터 인덱스의 값을 n_per만큼 추출(shot 개념)
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1) # batch라는 list에 각 합쳐진 각 클래스 데이터를 tensor 형식으로 stack
            yield batch