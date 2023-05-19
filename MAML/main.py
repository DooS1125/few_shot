import torch
import numpy as np
from Data import *
from MAML import *
from Model import *
from test import meta_test
from test import *
from utils import *

import wandb

hyperparameter_defaults = dict(shot = 5, way = 10, features = 'mel', win_length = 360, 
                               n_mels = 20, n_mfcc = 20, sr = 4000)

wandb.init(config=hyperparameter_defaults, project="few_shot")
config = wandb.config

if config.features == 'mel':
    run_name = "maml_" + str(config.shot) + 'shot_' + str(config.way) + 'way_' + str(config.features) + '_' + str(config.win_length) + '_'  + str(config.n_mels) + '_' + str(config.sr)
else:
    run_name = "maml_" + str(config.shot) + 'shot_' + str(config.way) + 'way_' + str(config.features) + '_' + str(config.win_length) + '_'  + str(config.n_mfcc) + '_' + str(config.sr)
wandb.run.name=run_name

if __name__ == '__main__':
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    maml = MAML(config=config, device=device, num_tasks=100, inner_steps=10)
    train_acc, train_loss, val_acc, val_loss = maml.meta_train(100)
        
    test_data = ESC_DataSet(config=config, set_name = 'test')
    test_set = test_data.task_set(num_tasks=100)
    
    test_loss, test_acc = meta_test(test_set, device, config)

wandb.run.summary['train_loss'] = min(train_loss)
wandb.run.summary['train_acc'] = max(train_acc)
wandb.run.summary['val_loss'] = min(val_loss)
wandb.run.summary['val_acc'] = max(val_acc)
wandb.run.summary['test_loss'] = test_loss
wandb.run.summary['test_acc'] = test_acc
