import random
import numpy as np
import torch
import torch.nn.functional as F
import os
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import librosa

n_classes=50
n_train=30
n_val=10
n_test=n_classes-n_train-n_val
seed=0

def train_test_split_class(n=seed, n_classes=n_classes, n_train=n_train, n_val=n_val):
    torch.manual_seed(n)           # Set seed for reproducibility
    classes = torch.randperm(n_classes)  # Returns random permutation of numbers 0 to 50
    if n_classes <= n_train + n_val:
        raise Exception("check number of class")
    else:
        train_classes, val_classes, test_classes = classes[:n_train], classes[n_train:n_train + n_val], classes[n_train + n_val:]
    return train_classes, val_classes, test_classes

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

## Make pairs
def create_pairs(data, digit_indices):
    x0_data = []
    x1_data = []
    label = []
    n = min([len(digit_indices[d]) for d in range(n_classes)]) - 1
    for d in range(n_classes): # for MNIST dataset: as we have 10 digits
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            x0_data.append(data[z1]) # Image Preprocessing Step
            x1_data.append(data[z2]) # Image Preprocessing Step
            label.append(0)
            inc = random.randrange(n_classes)
            dn = (d + inc) % n_classes
            while dn == d:
                inc = random.randrange(n_classes)
                dn = (d + inc) % n_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            x0_data.append(data[z1]) # Image Preprocessing Step
            x1_data.append(data[z2]) # Image Preprocessing Step
            label.append(1)

    x0_data = np.array(x0_data, dtype=np.float32) #[:10201]
    x0_data = np.transpose(x0_data, (0,2,3,1))
    x1_data = np.array(x1_data, dtype=np.float32) #[:10201]
    x1_data = np.transpose(x1_data, (0,2,3,1))
    label = np.array(label, dtype=np.int32)
    return x0_data, x1_data, label

## Extract feature
def extract_features(path):
    directory_lists=os.listdir(path)
    X=[]
    Y=[]
    count=0
    if ('.DS_Store' in directory_lists):
            directory_lists.remove('.DS_Store')
    for d in directory_lists:
        nest=os.listdir(path+"/"+d)
        if ('.DS_Store' in nest):
            nest.remove('.DS_Store')
        for f in nest:
            img = image.load_img(path+"/"+d+"/"+f, target_size=(224, 224))
            img_data = image.img_to_array(img)
            img_data = preprocess_input(img_data)
            img_data = np.expand_dims(img_data, axis=0)
            X.append(img_data)
            Y.append(count)
        count+=1
    X=np.array(X)
    y=np.array(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)         
    return X_train, X_test, y_train, y_test 

## Plot graph
def plot_loss(train_loss,name="train_loss.png"):
    plt.plot(train_loss, label="train loss")
    plt.legend()
    
def plot_mnist(numpy_all, numpy_labels,name="./embeddings_plot.png"):
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999', '#000fff']
        for i in range(0,3):
            f = numpy_all[np.where(numpy_labels == i)]
            plt.plot(f[:, 0], f[:, 1], '.', c=c[i])
        plt.legend(['0', '1', '2'])
        plt.savefig(name)




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        wrong_k = batch_size - correct_k
        res.append(wrong_k.mul_(100.0 / batch_size))

    return res
    # return res, pred[:1].squeeze(0)


def save_checkpoint(state, is_best, args):
    directory = args.log_dir
    filename = directory + f"/checkpoint_{state['epoch']}.pth"

    if not os.path.exists(directory):
        os.makedirs(directory)

    if is_best:
        filename = directory + "/model_best.pth"
        torch.save(state, filename)
    else:
        torch.save(state, filename)


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    img = img.to('cpu')
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def plot_classes_preds(preds, probs, images, labels, N_way, K_shot, K_shot_test):
    """
    Plot Prediction Samples.
    Parameters
    ----------
    preds : list
        contain prediction value range from 0 to N_way - 1.
    probs : list
        contain prediction probability range from 0.0 ~ 1.0 formed softmax.
    images : list
        images[0] is sample images. Shape is (N_way, K_shot, 1, 28, 28).
        images[1] is query images. Shape is (N_way, K_shot, 1, 28, 28).
    labels: list
        labels[0] contains y value for sample image.
        labels[1] contains y value for query image.
    """
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(30, 20))
    sample_images, query_images = images
    probs = [el[i].item() for i, el in zip(labels[1], probs)]

    # display sample images
    for row in np.arange(K_shot):
        for col in np.arange(N_way):
            ax = fig.add_subplot(2 * K_shot, N_way, row * N_way + col + 1, xticks=[], yticks=[])
            matplotlib_imshow(sample_images[col * K_shot + row], one_channel=True)
            ax.set_title(labels[0][col * K_shot + row].item())

    # display query images
    for row in np.arange(K_shot_test):
        for col in np.arange(N_way):
            ax = fig.add_subplot(2 * K_shot_test, N_way, N_way * K_shot_test + row * N_way + col + 1, xticks=[],
                                 yticks=[])
            matplotlib_imshow(query_images[col * K_shot_test + row], one_channel=True)
            ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
                preds[col * K_shot_test + row],
                probs[col * K_shot_test + row] * 100.0,
                labels[1][col * K_shot_test + row]),
                color=(
                    "green" if preds[col * K_shot_test + row] == labels[1][col * K_shot_test + row].item() else "red"))

    return fig
