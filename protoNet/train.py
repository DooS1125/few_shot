import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from Data import *
from Sampler import *
from Proto import *
from utils import *
import wandb

def validation(model, data_loader, device, config):
    model.eval()
    val_loss, val_acc = [], []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            data, _ = [_.to(device) for _ in batch]
            p = config.shot * config.way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(config.shot, config.way, -1).mean(dim=0)

            label = torch.arange(config.way).repeat(config.query).type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            val_loss.append(loss.item())
            val_acc.append(acc)

        return np.mean(val_loss), np.mean(val_acc)

def train(model, train_loader, val_loader, optimizer, lr_scheduler, device, config):

    early_stopping = EarlyStopping(patience=10, path='./weights/best.pt', verbose=True)
    train_acc, train_loss, val_acc, val_loss = [], [], [], []

    for epoch in range(1, config.max_epoch + 1):
        model.train()
        start_time = time.time()

        _train_loss_epoch, _train_acc_epoch = [], []

        for batch in tqdm(train_loader):
            data, _ = [_.to(device) for _ in batch]
            p = config.shot * config.way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(config.shot, config.way, -1).mean(dim=0)

            label = torch.arange(config.way).repeat(config.query).type(torch.cuda.LongTensor)

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

        val_loss_epoch, val_acc_epoch = validation(model, val_loader, device, config)

        finish_time = time.time()
        epoch_time = finish_time - start_time

        print(f'Epoch [{epoch}], Train Loss : [{train_loss_epoch:.4f}] Train ACC : [{train_acc_epoch:.4f}] Val Loss : [{val_loss_epoch:.4f}] Val ACC : [{val_acc_epoch:.4f}] Epoch_time : [{epoch_time:.0f}s]')
        
        wandb.log({"train_loss": train_loss_epoch, "train_acc": train_acc_epoch, "val_loss": val_loss_epoch, "val_acc": val_acc_epoch}, step=epoch)
        
        val_acc.append(val_acc_epoch)

        if lr_scheduler is not None:
            lr_scheduler.step(val_loss_epoch)

        early_stopping(val_loss_epoch, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    return train_acc, train_loss, val_acc, val_loss

def inference(model, test_loader, device, config):
    model.eval()
    test_loss, test_acc = [], []
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            data, _ = [_.to(device) for _ in batch]
            p = config.shot * config.way
            data_shot, data_query = data[:p], data[p:]

            proto = model(data_shot)
            proto = proto.reshape(config.shot, config.way, -1).mean(dim=0)

            label = torch.arange(config.way).repeat(config.query).type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(data_query), proto)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)

            test_loss.append(loss.item())
            test_acc.append(acc)

        test_loss = np.mean(test_loss)
        test_acc = np.mean(test_acc)

        finish_time = time.time()
        epoch_time = finish_time - start_time
        print(f'Test Loss : [{test_loss:.4f}] Test ACC : [{test_acc:.4f}] Epoch_time : [{epoch_time:.0f}s]')

    return test_loss, test_acc