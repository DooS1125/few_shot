import numpy as np
import torch
import librosa
from scipy.signal import butter, lfilter
import os
import shutil
import time
import matplotlib.pyplot as plt
import random

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

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def plot_results(train_loss, val_loss, train_acc, val_acc):
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    axes[0].grid()
    axes[0].plot(train_loss, label='train_loss')
    axes[0].plot(val_loss, label='val_loss')
    axes[0].set_xlabel('epoch')
    axes[0].set_ylabel('loss')
    axes[0].set_title("Loss", fontsize=25)
    axes[0].legend()

    axes[1].grid()
    axes[1].plot(train_acc, label='train_acc')
    axes[1].plot(val_acc, label='val_acc')
    axes[1].set_xlabel('epoch')
    axes[1].set_ylabel('accuracy')
    axes[1].set_title("Accuracy", fontsize=25)
    axes[1].legend()

    plt.show()



