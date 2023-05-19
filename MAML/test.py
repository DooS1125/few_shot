import torch
from torch import nn, optim
import numpy as np
from Data import *
from MAML import *
from Model import *

def meta_test(task_collection, device, config):
    test_loss, test_acc = [], []
    start_time = time.time()
    for i in range(len(task_collection)):
        cnn = CNN(config.way).to(device)
        cnn.load_state_dict(torch.load('./weights/best.pt'))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), lr=1e-2)
        num_epochs = 100
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
        
        test_acc.append(100 * correct / total)
        test_loss.append(loss.item())
        
    test_acc = np.array(test_acc).mean()
    test_loss = np.array(test_loss).mean()
    
    finish_time = time.time()
    epoch_time = finish_time - start_time
    
    print(f'Test Loss : [{test_loss:.4f}] Test ACC : [{test_acc:.4f}] Epoch_time : [{epoch_time:.0f}s]')

    return test_loss, test_acc