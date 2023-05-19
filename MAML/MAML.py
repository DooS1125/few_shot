import torch
from torch import nn, optim
from Model import *
from Data import *
import numpy as np
import copy
from tqdm import tqdm
import wandb

class MAML:
    def __init__(self, config, device, num_tasks,
                 alpha = 1e-3, beta = 1e-3, inner_steps=1):
        self.device = device
        self.model = CNN(config.way).to(self.device)
        self.num_tasks = num_tasks
        self.weights = list(self.model.parameters())
        self.beta = beta
        
        self.train_data = ESC_DataSet(config=config, set_name = 'train')
        self.train_collection = self.train_data.task_set(num_tasks=self.num_tasks)
        
        self.val_data = ESC_DataSet(config=config, set_name = 'val')
        self.val_collection = self.val_data.task_set(num_tasks=self.num_tasks)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, weight_decay=1e-4)
        self.inner_steps = inner_steps
        
    def inner_loop(self, dataloader):
       
        temp_weights = [w.clone() for w in self.weights]
        data = iter(dataloader).next()
        support, query, label = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
        for i in range(self.inner_steps):
            outputs = self.model.parameterised(support, temp_weights)
            loss = self.criterion(outputs, label)
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - self.beta * g for w, g in zip(temp_weights, grad)]
            
        cnn = copy.deepcopy(self.model)
        correct = 0
        total = 0
        with torch.no_grad():
            cnn.eval()
            outputs = cnn(query)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label.to(self.device)).sum().item()
        inner_acc = 100 * correct / total

        outputs = self.model.parameterised(query, temp_weights)    
        inner_loss = self.criterion(outputs, label)

        return inner_loss, inner_acc

    def meta_train(self, num_epochs):    
        
        early_stopping = EarlyStopping(patience=10, path='./weights/best.pt', verbose=True)  
        train_acc, train_loss, val_acc, val_loss = [], [], [], []
        
        for epoch in tqdm(range(num_epochs)):
            
            start_time = time.time()
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            outer_loss = 0
            outer_acc = 0
            for i in range(self.num_tasks):
                inner_loss, inner_acc = self.inner_loop(self.train_collection[i])
                outer_loss += inner_loss
                outer_acc += inner_acc
                
            avg_loss = outer_loss/self.num_tasks
            avg_loss.backward()
            self.optimizer.step()
            ll = avg_loss.item()
            train_loss.append(ll)
            
            avg_acc = outer_acc/self.num_tasks
            train_acc.append(avg_acc)
            
            val_loss_epoch, val_acc_epoch = self.meta_val(50)
            
            finish_time = time.time()
            epoch_time = finish_time - start_time
            
            wandb.log({"train_loss": ll, "train_acc": avg_acc,
                       "val_loss": val_loss_epoch, "val_acc": val_acc_epoch}, step=epoch)
            
            print(f'Epoch [{epoch}], Train Loss : [{ll:.4f}] Train ACC : [{avg_acc:.4f}] Val Loss : [{val_loss_epoch:.4f}] Val ACC : [{val_acc_epoch:.4f}] Epoch_time : [{epoch_time:.0f}s]')

            val_loss.append(val_loss_epoch)
            val_acc.append(val_acc_epoch)
                        
            early_stopping(val_loss_epoch, self.model)
                
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
        return train_acc, train_loss, val_acc, val_loss

    def meta_val(self, num_epoch):
        val_loss, val_acc = [], []
        for i in range(len(self.val_collection)):
            cnn = copy.deepcopy(self.model)
            criterion = nn.CrossEntropyLoss()
            opt = optim.Adam(cnn.parameters(), lr=1e-2)
            testloader = self.val_collection[i]
            data1, data2, label = iter(testloader).next()
            cnn.train() 
            for j in range(num_epoch):
                opt.zero_grad()
                outputs = cnn(data1.to(self.device))
                loss = criterion(outputs, label.to(self.device))
                loss.backward()
                opt.step()
                
            correct = 0
            total = 0
            with torch.no_grad():
                cnn.eval()  
                outputs = cnn(data2.to(self.device))
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label.to(self.device)).sum().item()
                
            val_loss.append(loss.item())
            val_acc.append(100 * correct / total)
            
        val_loss = np.array(val_loss).mean()
        val_acc = np.array(val_acc).mean()
        
        return val_loss, val_acc