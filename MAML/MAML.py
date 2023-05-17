import torch
from torch import nn, optim
from Model import *
from Data import *
import numpy as np
import copy
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MAML:
    def __init__(self, Net="CNN", num_tasks=300, Nway=5, Kshot=5, alpha = 1e-3, beta = 1e-3, inner_steps=1):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")    
        self.num_tasks = num_tasks
        if Net == 'CNN':
            self.model = CNN(Nway).to(device)
        else:
            self.model = MAML_Layer(Nway).to(device)
        self.weights = list(self.model.parameters())
        self.beta = beta
        
        self.train_data = ESC_DataSet(Nway=Nway, Kshot=Kshot, set_name = 'train')
        self.train_collection = self.train_data.task_set(num_tasks=self.num_tasks)
        
        self.val_data = ESC_DataSet(Nway=Nway, Kshot=Kshot, set_name = 'val')
        self.val_collection = self.val_data.task_set(num_tasks=30)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha, weight_decay=1e-4)
        self.inner_steps = inner_steps

    def inner_loop(self, dataloader):
       
        temp_weights = [w.clone() for w in self.weights]
        data = iter(dataloader).next()
        support, query, label = data[0].to(device), data[1].to(device), data[2].to(device)
        for i in range(self.inner_steps):
            outputs = self.model.parameterised(support, temp_weights)
            loss = self.criterion(outputs, label)
            grad = torch.autograd.grad(loss, temp_weights)
            temp_weights = [w - self.beta * g for w, g in zip(temp_weights, grad)]
                  
        outputs = self.model.parameterised(query, temp_weights)    
        inner_loss = self.criterion(outputs, label)

        return inner_loss

    def meta_train(self, num_epochs):      
        loss_list = []
        acc = 0
        for epoch in tqdm(range(num_epochs), leave=True):
            torch.cuda.empty_cache()
            self.optimizer.zero_grad()
            outer_loss = 0
            #self.weights = list(self.model.parameters())
           
            for i in range(self.num_tasks):
                outer_loss += self.inner_loop(self.train_collection[i])
                
            avg_loss = outer_loss/self.num_tasks
            avg_loss.backward()
            self.optimizer.step()
            ll = avg_loss.item()
            loss_list.append(ll)
            
            val_acc = self.meta_val()
            if epoch % 10 == 0:
                print('[%d] Train loss: %.3f, Validation accuracy: %.2f %%' %(epoch, ll, val_acc))
             
            if val_acc >= acc:
                acc = val_acc
                print('Saved the model - Validation accuracy: %.2f %%' % (val_acc))   
                torch.save(self.model.state_dict(), './maml.pth') 
          
        return loss_list

    def meta_val(self):

        acc_list = []
        for i in range(len(self.val_collection)):
            cnn = copy.deepcopy(self.model)
            criterion = nn.CrossEntropyLoss()
            opt = optim.Adam(cnn.parameters(), lr=1e-2)
            num_epochs = 101
            testloader = self.val_collection[i]
            data1, data2, label = iter(testloader).next()
            cnn.train() 
            for j in range(num_epochs):
                opt.zero_grad()
                outputs = cnn(data1.to(device))
                loss = criterion(outputs, label.to(device))
                loss.backward()
                opt.step()
                
            correct = 0
            total = 0
            with torch.no_grad():
                cnn.eval()  
                outputs = cnn(data2.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += label.size(0)
                correct += (predicted == label.to(device)).sum().item()
            
            acc_list.append(100 * correct / total)
            
        acc_info = np.array(acc_list)
        avg_acc = np.mean(acc_info)
        return avg_acc