import os.path as osp
from pprint import pprint

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm 

from Data import *
from Sampler import *
from Proto import *
from utils import *
from train import *

import wandb

hyperparameter_defaults = dict(max_epoch = 100, query = 5, test_batch = 100,
                               shot = 5, way = 5, features = 'mel', win_length = 360, n_mels = 20, n_mfcc = 14, sr = 4000)

wandb.init(config=hyperparameter_defaults, project="23_1_seminar")
config = wandb.config

if config.features == 'mel':
    run_name = str(config.shot) + 'shot_' + str(config.way) + 'way_' + str(config.features) + '_' + str(config.win_length) + '_'  + str(config.n_mels) + '_' + str(config.sr)
else:
    run_name = str(config.shot) + 'shot_' + str(config.way) + 'way_' + str(config.features) + '_' + str(config.win_length) + '_'  + str(config.n_mfcc) + '_' + str(config.sr)
wandb.run.name=run_name

if __name__ == '__main__':
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if config.features == 'mel':
        trainset = ESC_data('train', win_length = config.win_length, n_mels = config.n_mels, n_mfcc = config.n_mfcc, feature = config.features, sample_rate = config.sr)
        train_sampler = CategoriesSampler(trainset.label, 100, config.way, config.shot + config.query)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)

        valset = ESC_data('val', win_length = config.win_length, n_mels = config.n_mels, n_mfcc = config.n_mfcc, feature = config.features, sample_rate = config.sr)
        val_sampler = CategoriesSampler(valset.label, 50, config.way, config.shot + config.query)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=0, pin_memory=True)
    else:
        trainset = ESC_data('train', win_length = config.win_length, n_mels = config.n_mels, n_mfcc = config.n_mfcc, feature = config.features, sample_rate = config.sr)
        train_sampler = CategoriesSampler(trainset.label, 100, config.way, config.shot + config.query)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)

        valset = ESC_data('val', win_length = config.win_length, n_mels = config.n_mels, n_mfcc = config.n_mfcc, feature = config.features, sample_rate = config.sr)
        val_sampler = CategoriesSampler(valset.label, 50, config.way, config.shot + config.query)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=0, pin_memory=True)


    model = Convnet().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, 
                                                              threshold_mode='abs', min_lr=1e-5, verbose=True)
    
    train_acc, train_loss, val_acc, val_loss = train(model, train_loader, val_loader, optimizer, lr_scheduler, device, config)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if config.features == 'mel':
        dataset = ESC_data('test', win_length = config.win_length, n_mels = config.n_mels, n_mfcc = config.n_mfcc, feature = config.features, sample_rate = config.sr)
        sampler = CategoriesSampler(dataset.label, config.test_batch, config.way, config.shot + config.query)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)
    else:
        dataset = ESC_data('test', win_length = config.win_length, n_mels = config.n_mels, n_mfcc = config.n_mfcc, feature = config.features, sample_rate = config.sr)
        sampler = CategoriesSampler(dataset.label, config.test_batch, config.way, config.shot + config.query)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)

    model = Convnet().to(device)
    model.load_state_dict(torch.load('./weights/best.pt'))
    
    test_loss, test_acc = inference(model, loader, device, config)


wandb.run.summary['train_loss'] = train_loss[config.max_epoch-1]
wandb.run.summary['train_acc'] = train_acc[config.max_epoch-1]
wandb.run.summary['val_loss'] = val_loss[config.max_epoch-1]
wandb.run.summary['val_acc'] = val_acc[config.max_epoch-1]
wandb.run.summary['test_loss'] = test_loss
wandb.run.summary['test_acc'] = test_acc