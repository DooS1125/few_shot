import torch
import pandas as pd
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
import os.path as osp
from utils import *
import librosa
from random import Random

csv_path = 'C:/Users/ParkDooseo/Desktop/few_shot/ESC_data/esc50.csv'
ROOT_PATH = 'C:/Users/ParkDooseo/Desktop/few_shot/ESC_data/alldata'
train_classes, val_classes, test_classes = train_test_split_class()
n_mels=40

class ESC_data(data.Dataset):

    def __init__(self, feature='mel', sample_rate=22050):
        
        data_csv = pd.read_csv(csv_path)
        class_csv = data_csv[['filename','target','category']]
        
        class_mask = (torch.from_numpy(class_csv['target'].values)[:, None] == train_classes[None,:]).any(dim=-1)
        filename_mask=class_csv.values[class_mask]
        new_df=pd.DataFrame(filename_mask, columns=['filename','target','category']).sort_values(by='target').reset_index(drop=True)
        data = []
        label = []
        
        lb=-1
        
        self.category=[]
        
        for index, row in new_df.iterrows():
            # category list로 category를 append 하고 새로운 category 나올 시 lb값을 1 증가시키는 구조 / 이렇게 해서 각 데이터 셋(train, val, test)의 label 값을 0~해당 클래스 수로 초기화
            # self.category에는 해당 클래스의 실제 category 있음
            path = osp.join(ROOT_PATH, row.filename)
            if row.category not in self.category:
                self.category.append(row.category) 
                lb +=1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        self.sample_rate=sample_rate
        self.feature=feature
        self.pair = [self.data, self.label]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        
        if i % 2 == 1:
            pair_label = 1.0
            idx = random.randint(0, len(self.label) - 1)
            data_list = [x for x in self.pair if x[1] == idx]
            data1 = random.choice(data_list)
            data2 = random.choice(data_list)
            while data1[0] == data2[0]:
                data2 = random.choice(data_list)
        # get image from different class
        else:
            pair_label = 0.0
            data1 = random.choice(self.pair)
            data2 = random.choice(self.pair)
            while data1[1] == data2[1]:
                data2 = random.choice(self.pair)   
                
        data1_sam, _ = librosa.load(path=data1[0], sr=self.sample_rate)
        data2_sam, _ = librosa.load(path=data2[0], sr=self.sample_rate)
        data1_sam = norm_max(data1_sam)
        data2_sam = norm_max(data2_sam)

        if self.feature=='mel':
            audio1 = log_mel(x=data1_sam, n_mel=n_mels, sr=self.sample_rate, n_fft=372, hop_length=93)
            audio2 = log_mel(x=data2_sam, n_mel=n_mels, sr=self.sample_rate, n_fft=372, hop_length=93)
        elif self.feature == 'mfcc':
            audio1 = MFCC(x=data1_sam, n_fft=372, win_length=372, hop_length=93, sr=self.sample_rate, n_mfcc=14)
            audio2 = MFCC(x=data2_sam, n_fft=372, win_length=372, hop_length=93, sr=self.sample_rate, n_mfcc=14)
        audio1 = torch.Tensor(audio1).unsqueeze(0)
        audio2 = torch.Tensor(audio2).unsqueeze(0)
        pair_label = torch.from_numpy(np.array(pair_label, dtype=np.float32))

        return audio1, audio2, pair_label
    
    
    
class ESC_testdata(data.Dataset):

    def __init__(self, set_name, way, trials, feature='mel', seed=0, sample_rate=22050 ):
        
        data_csv = pd.read_csv(csv_path)
        class_csv = data_csv[['filename','target','category']]
        
        if set_name =='val':
            class_mask = (torch.from_numpy(class_csv['target'].values)[:, None] == val_classes[None,:]).any(dim=-1)
            filename_mask=class_csv.values[class_mask]
            new_df=pd.DataFrame(filename_mask, columns=['filename','target','category']).sort_values(by='target').reset_index(drop=True)
        elif set_name =='test':
            class_mask = (torch.from_numpy(class_csv['target'].values)[:, None] == test_classes[None,:]).any(dim=-1)
            filename_mask=class_csv.values[class_mask]
            new_df=pd.DataFrame(filename_mask, columns=['filename','target','category']).sort_values(by='target').reset_index(drop=True)


        data = []
        label = []
        
        lb=-1
        
        self.category=[]
        
        for index, row in new_df.iterrows():
            # category list로 category를 append 하고 새로운 category 나올 시 lb값을 1 증가시키는 구조 / 이렇게 해서 각 데이터 셋(train, val, test)의 label 값을 0~해당 클래스 수로 초기화
            # self.category에는 해당 클래스의 실제 category 있음
            path = osp.join(ROOT_PATH, row.filename)
            if row.category not in self.category:
                self.category.append(row.category) 
                lb +=1
            data.append(path)
            label.append(lb)

        self.way = way
        self.trials = trials
        self.seed = seed
        self.data1 = None

        self.data = data
        self.label = label
        self.sample_rate=sample_rate
        self.feature=feature
        self.pair = [self.data, self.label]

    def __len__(self):
        return self.trials * self.way

    def __getitem__(self, i):
        rand = Random(self.seed + i)
        if i % 2 == 1:
            pair_label = 1.0
            idx = random.randint(0, len(self.label) - 1)
            data_list = [x for x in self.pair if x[1] == idx]
            self.data1 = random.choice(data_list)
            data2 = random.choice(data_list)
            while self.data1[0] == data2[0]:
                data2 = random.choice(data_list)
        # get image from different class
        else:
            pair_label = 0.0
            self.data1 = random.choice(self.pair)
            data2 = random.choice(self.pair)
            while self.data1[1] == data2[1]:
                data2 = random.choice(self.pair)   
                
        data1_sam, _ = librosa.load(path=self.data1[0], sr=self.sample_rate)
        data2_sam, _ = librosa.load(path=data2[0], sr=self.sample_rate)
        data1_sam = norm_max(data1_sam)
        data2_sam = norm_max(data2_sam)

        if self.feature=='mel':
            audio1 = log_mel(x=data1_sam, n_mel=n_mels, sr=self.sample_rate, n_fft=372, hop_length=93)
            audio2 = log_mel(x=data2_sam, n_mel=n_mels, sr=self.sample_rate, n_fft=372, hop_length=93)
        elif self.feature == 'mfcc':
            audio1 = MFCC(x=data1_sam, n_fft=372, win_length=372, hop_length=93, sr=self.sample_rate, n_mfcc=14)
            audio2 = MFCC(x=data2_sam, n_fft=372, win_length=372, hop_length=93, sr=self.sample_rate, n_mfcc=14)
        audio1 = torch.Tensor(audio1).unsqueeze(0)
        audio2 = torch.Tensor(audio2).unsqueeze(0)
        pair_label = torch.from_numpy(np.array(pair_label, dtype=np.float32))

        return audio1, audio2, pair_label

        
        
        
def get_train_validation_loader(batch_size, way, trials, shuffle, seed, num_workers, pin_memory):
    
    train_dataset = ESC_data()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    val_dataset = ESC_testdata(set_name='val', way=way, trials=trials, seed=seed)
    val_loader = DataLoader(val_dataset, batch_size=way, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


def get_test_loader(way, trials, seed, num_workers, pin_memory):
    test_dataset = ESC_testdata(set_name='test', way=way, trials=trials, seed=seed)
    test_loader = DataLoader(test_dataset, batch_size=way, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    return test_loader
