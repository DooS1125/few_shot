import numpy as np
import torch
import librosa
from scipy.signal import butter, lfilter
import os
import shutil
import time

# Transform Class
# Preprocessing Def
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5*fs
    low=lowcut/nyq
    high=highcut/nyq
    b, a= butter(order, [low, high], btype='band')
    return b,a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, duration=None):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def norm_max(x):
    x = x / (np.max(x))
    return x


### Feature Extraction
def log_mel(x, n_mel=14, sr=4000, n_fft=372, hop_length=93):
    S =  librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mel, n_fft=n_fft, hop_length=hop_length) # 320/80
    log_S = librosa.amplitude_to_db(S**(1/2))
    return log_S

def MFCC(x, n_fft=372, win_length=372, hop_length=93, sr=4000, n_mfcc=14):
    D = np.abs(librosa.stft(y=x, n_fft = n_fft, win_length = win_length, hop_length = hop_length))
    mfcc = librosa.feature.mfcc(S = librosa.power_to_db(D), sr = sr, n_mfcc = n_mfcc)
    return mfcc




def split_batch(audios, targets):
    support_audios, query_audios = audios.chunk(2, dim=0)
    support_targets, query_targets = targets.chunk(2, dim=0)
    return support_audios, query_audios, support_targets, query_targets


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)

def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)

class Averager():
    def __init__(self):
        self.n = 0
        self.v = 0
    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1
    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()

def dot_metric(a, b):
    return torch.mm(a, b.t())

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

class Timer():
    def __init__(self):
        self.o = time.time()
    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2



def train_test_split_class(n=0, n_classes=50, n_train=30, n_val=10):
    torch.manual_seed(n)           # Set seed for reproducibility
    classes = torch.randperm(n_classes)  # Returns random permutation of numbers 0 to 50
    if n_classes <= n_train + n_val:
        raise Exception("check number of class")
    else:
        train_classes, val_classes, test_classes = classes[:n_train], classes[n_train:n_train + n_val], classes[n_train + n_val:]
    return train_classes, val_classes, test_classes


class EarlyStopping:
    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path='./weights/checkpoint.pt'):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model, mode = True):

        score = -val_loss
        if mode:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
        else:
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score >= self.best_score + self.delta:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss