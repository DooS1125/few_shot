import torch
import numpy as np
from Data import *
from MAML import *
from Model import *
from test import *
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

Nway = 5
Kshot = 5

maml = MAML(num_tasks=10, Net = 'CNN', Nway=Nway, Kshot=Kshot)
loss = maml.meta_train(10)

test_data = ESC_DataSet(Nway=Nway, Kshot=Kshot, set_name = 'test')
test_set = test_data.task_set(num_tasks=100)

meta_test(test_set, Net='CNN',pretrained='meta')

meta_test(test_set)