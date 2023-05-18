import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import numpy as np
from utils import *
import pandas as pd
import os.path as osp

#csv_path = 'C:/Users/ParkDooseo/Desktop/few_shot/ESC_data/esc50.csv'
csv_path = '../ESC_data/esc50.csv'
ROOT_PATH = 'C:/Users/ParkDooseo/Desktop/few_shot/ESC_data/alldata'
train_classes, val_classes, test_classes = train_test_split_class()
n_mels=20

class ESC_data(Dataset):

    def __init__(self, support_set, query_set, labels, feature='mel', sample_rate=22050 ):
        
        self.support_set = torch.FloatTensor(support_set)
        self.query_set = torch.FloatTensor(query_set)
        self.labels = torch.LongTensor(labels)
        self.len = self.labels.shape[0]
                
    def __getitem__(self, index):
        sample = self.support_set[index], self.query_set[index], self.labels[index]
        return sample

    def __len__(self):
        return self.len


class ESC_DataSet:
    
    def __init__(self, Nway, Kshot, set_name, feature='mel', sample_rate=8000 ):
        
        self.Nway = Nway
        self.Kshot = Kshot
        
        data_csv = pd.read_csv(csv_path)
        class_csv = data_csv[['filename','target','category']]

        if set_name == 'train':
            class_mask = (torch.from_numpy(class_csv['target'].values)[:, None] == train_classes[None,:]).any(dim=-1)
            filename_mask=class_csv.values[class_mask]
            new_df=pd.DataFrame(filename_mask, columns=['filename','target','category']).sort_values(by='target').reset_index(drop=True)
        elif set_name =='val':
            class_mask = (torch.from_numpy(class_csv['target'].values)[:, None] == val_classes[None,:]).any(dim=-1)
            filename_mask=class_csv.values[class_mask]
            new_df=pd.DataFrame(filename_mask, columns=['filename','target','category']).sort_values(by='target').reset_index(drop=True)
        elif set_name =='test':
            class_mask = (torch.from_numpy(class_csv['target'].values)[:, None] == test_classes[None,:]).any(dim=-1)
            filename_mask=class_csv.values[class_mask]
            new_df=pd.DataFrame(filename_mask, columns=['filename','target','category']).sort_values(by='target').reset_index(drop=True)

        data = []
        label = []
        
        self.category=[]
        self.cls=[]
        for index, row in new_df.iterrows():
            path = osp.join(ROOT_PATH, row.filename)
            if row.category not in self.category:
                self.category.append(row.category) 
                self.cls.append(row.target)
            data.append(path)
            label.append(row.target)
            

        self.data = data
        self.label = label
        self.sample_rate=sample_rate
        self.feature=feature

    def task_generator(self):
        
        labels = np.array(self.label)
        
        support_set = []
        query_set = []
        targets = []
        for i, c in enumerate(self.cls):
            idx = np.where(labels == c)[0]
            rand = np.random.choice(len(idx), 2*self.Kshot, replace=False)
            for shot in range(2*self.Kshot):
                audio_path = self.data[idx[rand[shot]]]
                if shot < self.Kshot:
                    samples, _ = librosa.load(path=audio_path, sr=self.sample_rate)
                    samples = norm_max(samples)
                    if self.feature=='mel':
                        audio = log_mel(x=samples, n_mel=n_mels, sr=self.sample_rate, n_fft=372, hop_length=93)
                    elif self.feature == 'mfcc':
                        audio = MFCC(x=samples, n_fft=372, win_length=372, hop_length=93, sr=self.sample_rate, n_mfcc=14)
                    audio = torch.Tensor(audio).unsqueeze(0)
                    support_set.append(audio)
                    targets.append(i)
                else:
                    samples, _ = librosa.load(path=audio_path, sr=self.sample_rate)
                    samples = norm_max(samples)
                    if self.feature=='mel':
                        audio = log_mel(x=samples, n_mel=n_mels, sr=self.sample_rate, n_fft=372, hop_length=93)
                    elif self.feature == 'mfcc':
                        audio = MFCC(x=samples, n_fft=372, win_length=372, hop_length=93, sr=self.sample_rate, n_mfcc=14)
                    audio = torch.Tensor(audio).unsqueeze(0)
                    query_set.append(audio)  

        support_set = np.stack(support_set)
        query_set = np.stack(query_set)
        targets = np.array(targets)
        taskset = ESC_data(support_set, query_set, targets)
        dataloader = torch.utils.data.DataLoader(taskset, batch_size=self.Nway*self.Kshot, shuffle=False)
        return dataloader

    def task_set(self, num_tasks):
        task_collection = []
        for i in range(num_tasks):
            task_collection.append(self.task_generator())
        return task_collection