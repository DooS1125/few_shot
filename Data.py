import torch
import torch.utils.data as data
from torchvision import transforms
import os
import os.path as osp
import pandas as pd
from utils.utils import *

### train, val, test 나누는 코드 필요?
csv_path = './ESC_data/esc50.csv'
ROOT_PATH = './ESC_data/alldata/'
train_classes, val_classes, test_classes = train_test_split_class()
n_mels=40

class ESC_data(data.Dataset):
    def __init__(self, set_name, feature='mel', sample_rate=22050):
        """
        class_to_idx = data_csv[['target', 'category']].drop_duplicates().sort_values(by='target').set_index('category').to_dict()['target']
        category_df = data_csv[['category','target']].sort_values(by='target').drop_duplicates(subset='target').reset_index(drop=True)
        classes = category_df['category'].to_list()
        """
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


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        samples, _ = librosa.load(path=path, sr=self.sample_rate)
        samples = norm_max(samples)

        if self.feature=='mel':
            audio = log_mel(x=samples, n_mel=n_mels, sr=self.sample_rate, n_fft=372, hop_length=93)
        elif self.feature == 'mfcc':
            audio = MFCC(x=samples, n_fft=372, win_length=372, hop_length=93, sr=self.sample_rate, n_mfcc=14)
        
        audio = torch.Tensor(audio).unsqueeze(0)
        
        return audio, label
