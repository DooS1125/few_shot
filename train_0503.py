import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from Data import *
from Sampler import *
from models.Proto import *
from utils.utils import *
from utils.early_stopping import EarlyStopping

def validation(model, data_loader, device, args):
    model.eval()
    val_loss, val_acc = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            data, _ = [_.to(device) for _ in batch]
            p = args.shot * args.test_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.test_way, -1).mean(dim=0)

            label = torch.arange(args.test_way).repeat(args.query).type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            val_loss.append(loss.item())
            val_acc.append(acc)

        return np.mean(val_loss), np.mean(val_acc)

def train(model, train_loader, val_loader, optimizer, lr_scheduler, device, args):

    early_stopping = EarlyStopping(patience=5, path='./weights/best.pt', verbose=True)
    train_acc, train_loss, val_acc, val_loss = [], [], [], []

    for epoch in range(1, args.max_epoch + 1):
        model.train()
        start_time = time.time()

        _train_loss_epoch, _train_acc_epoch = [], []

        for batch in tqdm(train_loader):
            data, _ = [_.to(device) for _ in batch]
            p = args.shot * args.train_way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(args.shot, args.train_way, -1).mean(dim=0)

            label = torch.arange(args.train_way).repeat(args.query).type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _train_acc_epoch.append(acc)
            _train_loss_epoch.append(loss.item())

            train_acc_epoch = np.mean(_train_acc_epoch)
            train_loss_epoch = np.mean(_train_loss_epoch)

        train_acc.append(train_acc_epoch)
        train_loss.append(train_loss_epoch)

        val_loss_epoch, val_acc_epoch = validation(model, val_loader, device, args)

        finish_time = time.time()
        epoch_time = finish_time - start_time

        print(f'Epoch [{epoch}], Train Loss : [{train_loss_epoch:.4f}] Train ACC : [{train_acc_epoch:.4f}] Val Loss : [{val_loss_epoch:.4f}] Val ACC : [{val_acc_epoch:.4f}] Epoch_time : [{epoch_time:.0f}s]')

        val_loss.append(val_loss_epoch)
        val_acc.append(val_acc_epoch)

        if lr_scheduler is not None:
            lr_scheduler.step(val_loss_epoch)

        early_stopping(val_loss_epoch, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_acc, train_loss, val_acc, val_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=2)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--train-way', type=int, default=5)
    parser.add_argument('--test-way', type=int, default=5)
    args = parser.parse_args()

    seed_everything(2023)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    trainset = ESC_data('train')
    train_sampler = CategoriesSampler(trainset.label, 30, 
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                        num_workers=0, pin_memory=True)

    valset = ESC_data('val')
    val_sampler = CategoriesSampler(valset.label, 30, 
                                    args.test_way, args.shot + args.query)
    val_loader = DataLoader(valset, batch_sampler=val_sampler, 
                            num_workers=0, pin_memory=True)

    model = Convnet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, 
                                                            threshold_mode='abs', min_lr=1e-5, verbose=True)

    train_acc, train_loss, val_acc, val_loss = train(model, train_loader, val_loader, optimizer, lr_scheduler, device, args)

    plot_results(train_loss, val_loss, train_acc, val_acc)
