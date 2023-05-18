import argparse
import os.path as osp
from pprint import pprint

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Data import *
from Sampler import *
from Proto import *
from utils import *
from tqdm import tqdm 

import wandb

save_path = './save'
save_epoch = 20
gpu = '0'
test_batch = 100
load_path = './save/max-acc.pth'

hyperparameter_defaults = dict(max_epoch = 100, query = 5, 
                               shot = 5, way = 5, features = 'mel', win_length = 360, n_mels = 20, n_mfcc = 14, sr = 4000)

wandb.init(config=hyperparameter_defaults, project="23_1_seminar")
config = wandb.config

if config.features == 'mel':
    run_name = str(config.shot) + 'shot_' + str(config.way) + 'way_' + str(config.features) + '_' + str(config.win_length) + '_'  + str(config.n_mels) + '_' + str(config.sr)
else:
    run_name = str(config.shot) + 'shot_' + str(config.way) + 'way_' + str(config.features) + '_' + str(config.win_length) + '_'  + str(config.n_mfcc) + '_' + str(config.sr)
wandb.run.name=run_name

if __name__ == '__main__':

    set_gpu(gpu)
    ensure_path(save_path)
    if config.features == 'mel':
        trainset = ESC_data('train', win_length = config.win_length, n_mels = config.n_mels, n_mfcc = config.n_mfcc, feature = config.features, sample_rate = config.sr)
        train_sampler = CategoriesSampler(trainset.label, 100,
                                        config.way, config.shot + config.query)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                  num_workers=0, pin_memory=True)

        valset = ESC_data('val', win_length = config.win_length, n_mels = config.n_mels, n_mfcc = config.n_mfcc, feature = config.features, sample_rate = config.sr)
        val_sampler = CategoriesSampler(valset.label, 50,
                                        config.way, config.shot + config.query)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=0, pin_memory=True)
    else:
        trainset = ESC_data('train', win_length = config.win_length, n_mels = config.n_mels, n_mfcc = config.n_mfcc, feature = config.features, sample_rate = config.sr)
        train_sampler = CategoriesSampler(trainset.label, 100,
                                        config.way, config.shot + config.query)
        train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                                  num_workers=0, pin_memory=True)

        valset = ESC_data('val', win_length = config.win_length, n_mels = config.n_mels, n_mfcc = config.n_mfcc, feature = config.features, sample_rate = config.sr)
        val_sampler = CategoriesSampler(valset.label, 50,
                                        config.way, config.shot + config.query)
        val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                                num_workers=0, pin_memory=True)


    model = Convnet().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        torch.save(model.state_dict(), osp.join(save_path, name + '.pth'))
    
    train_loss=[]
    val_loss=[]
    train_acc=[]
    val_acc=[]
    max_acc=0.0

    timer = Timer()

    
    for epoch in tqdm(range(1, config.max_epoch + 1), leave=True):

        model.train()
        
        tl = Averager()
        ta = Averager()
        
        for i, batch in tqdm(enumerate(train_loader, 1), leave=True):
            data, _ = [_.cuda() for _ in batch]
            p = config.shot * config.way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(config.shot, config.way, -1).mean(dim=0)

            label = torch.arange(config.way).repeat(config.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            # print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
            #       .format(epoch, i, len(train_loader), loss.item(), acc))

            tl.add(loss.item())
            ta.add(acc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            proto = None; logits = None; loss = None
            
        lr_scheduler.step()

        tl = tl.item()
        ta = ta.item()

        model.eval()

        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = config.shot * config.way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(config.shot, config.way, -1).mean(dim=0)

            label = torch.arange(config.way).repeat(config.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            vl.add(loss.item())
            va.add(acc)
            
            proto = None; logits = None; loss = None

        vl = vl.item()
        va = va.item()
        print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, vl, va))

        if va > max_acc:
            max_acc = va
            save_model('max-acc')

        train_loss.append(tl)
        train_acc.append(ta)
        val_loss.append(vl)
        val_acc.append(va)
        
        save_model('epoch-last')

        if epoch % save_epoch == 0:
            save_model('epoch-{}'.format(epoch))

        print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / config.max_epoch)))


if __name__ == '__main__':
    set_gpu(gpu)
    if config.features == 'mel':
        dataset = ESC_data('test', win_length = config.win_length, n_mels = config.n_mels, n_mfcc = config.n_mfcc, feature = config.features, sample_rate = config.sr)
        sampler = CategoriesSampler(dataset.label, test_batch, config.way, config.shot + config.query)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)
    else:
        dataset = ESC_data('test', win_length = config.win_length, n_mels = config.n_mels, n_mfcc = config.n_mfcc, feature = config.features, sample_rate = config.sr)
        sampler = CategoriesSampler(dataset.label, test_batch, config.way, config.shot + config.query)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)

    model = Convnet().cuda()
    model.load_state_dict(torch.load(load_path))
    model.eval()

    ave_acc = Averager()

    for i, batch in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in batch]
        k = config.way * config.shot
        data_shot, data_query = data[:k], data[k:]

        x = model(data_shot)
        x = x.reshape(config.shot, config.way, -1).mean(dim=0)
        p = x

        logits = euclidean_metric(model(data_query), p)

        label = torch.arange(config.way).repeat(config.query)
        label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        
        x = None; p = None; logits = None


wandb.run.summary['train_loss'] = tl
wandb.run.summary['train_acc'] = ta
wandb.run.summary['val_loss'] = vl
wandb.run.summary['val_loss'] = va
wandb.run.summary['ave_acc'] = acc