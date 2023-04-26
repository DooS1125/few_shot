import argparse
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--save-epoch', type=int, default=20)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    parser.add_argument('--save-path', default='./save')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()
    # pprint(vars(args))

    # set_gpu(args.gpu)
    ensure_path(args.save_path)

    trainset = ESC_data('train')
    train_sampler = CategoriesSampler(trainset.label, 30,
                                      args.train_way, args.shot + args.query)

    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=0, pin_memory=True)

    valset = ESC_data('val')
    val_sampler = CategoriesSampler(valset.label, 30,
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                            num_workers=0, pin_memory=True)

    model = Convnet().cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    def save_model(name):
        print('weight saved...')
        torch.save(model.state_dict(), osp.join(args.save_path, name + '.pth'))
    
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['train_acc'] = []
    trlog['val_acc'] = []
    trlog['max_acc'] = 0.0

    timer = Timer()

    def validate(epoch, model, val_loader, args):
        model.eval()
        vl = Averager()
        va = Averager()

        for i, batch in enumerate(val_loader, 1):
            data, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            print('epoch {}, valid {}/{}, loss={:.4f} acc={:.4f}'
                .format(epoch, i, len(val_loader), loss.item(), acc))

            vl.add(loss.item())
            va.add(acc)

            proto = None; logits = None; loss = None

        return vl.item(), va.item()

    def train(model, train_loader, val_loader, optimizer, lr_scheduler, args):
        for epoch in range(1, args.max_epoch + 1):
            model.train()
            tl = Averager()
            ta = Averager()

            for i, batch in enumerate(train_loader, 1):
                data, _ = [_.cuda() for _ in batch]
                p = args.shot * args.train_way
                data_shot, data_query = data[:p], data[p:]

                proto = model(data_shot)
                proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

                label = torch.arange(args.train_way).repeat(args.query)
                label = label.type(torch.cuda.LongTensor)

                logits = euclidean_metric(model(data_query), proto)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'
                    .format(epoch, i, len(train_loader), loss.item(), acc))

                tl.add(loss.item())
                ta.add(acc)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                proto = None; logits = None; loss = None

            lr_scheduler.step()
            train_loss, train_acc = tl.item(), ta.item()

            val_loss, val_acc = validate(epoch, model, val_loader, args)
            print('epoch {}, val, loss={:.4f} acc={:.4f}'.format(epoch, val_loss, val_acc))

            if epoch == args.max_epoch:
                print('epoch {}, train_loss={:.4f}, train_acc={:.4f}, val_loss={:.4f}, val_acc={:.4f}'.format(epoch, train_loss, train_acc, val_loss, val_acc))

        return train_loss, train_acc, val_loss, val_acc


    train_loss, train_acc, val_loss, val_acc = train(model, train_loader, val_loader, optimizer, lr_scheduler, args)
