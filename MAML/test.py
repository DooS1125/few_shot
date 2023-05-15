import torch
from torch import nn, optim
import numpy as np
from Data import *
from MAML import *
from Model import *

Nway=5

def meta_test(task_collection, pretrained=None):

    acc_list = []
    for i in range(len(task_collection)):
        cnn = CNN(Nway).to(device)
        if pretrained == 'meta':
            cnn.load_state_dict(torch.load('./models/cifar_maml.pth'))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), lr=1e-2)
        num_epochs = 101
        testloader = task_collection[i]
        data1, data2, label = iter(testloader).next()
        cnn.train() 
        for j in range(num_epochs):
            optimizer.zero_grad()
            outputs = cnn(data1.to(device))
            loss = criterion(outputs, label.to(device))
            loss.backward()
            optimizer.step()
            
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
    print('Avg. Test accuracy: %.2f %% ± %.2f' % (np.mean(acc_info), 1.96*np.std(acc_info)/np.sqrt(len(acc_info))))   