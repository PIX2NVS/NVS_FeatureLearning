from __future__ import division
import os.path as osp
import numpy as np
import math
import shutil
import glob
import os
import random
from utils import Logger

import torch
from torch.utils.data.dataset import Subset
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from inputsdata import MyOwnDataset
import torch_geometric.transforms as T
from torch.utils.data import Dataset#, DataLoader
from torch_geometric.data import DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x, graclus, global_mean_pool, NNConv
from MyNet import Net

def parse_files(train_filename, filenames):
    print('Parsing inputs: ')
    with open(train_filename) as f:
        files_to_parse = [os.path.splitext(line.split()[0].split('/')[1])[0] for line in f]
        filenames = [d for d in filenames for item in files_to_parse if item in d]  
        return filenames

    
    
def train(epoch, batch_logger, train_loader):
    model.train()
    if epoch == 60:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001
    if epoch == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.00001
    for i, sample in enumerate(train_loader):
        with autograd.detect_anomaly():
            sample = sample.to(device)      #############
            
            optimizer.zero_grad()
            end_point = model(sample)
            
            print(sample.y)
            
            loss = F.nll_loss(end_point, sample.y)
            pred = end_point.max(1)[1]
            acc = (pred.eq(sample.y).sum().item())/len(sample.y)
            
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                batch_logger.log({'epoch': epoch,'batch': i + 1,'loss': loss.item(),'acc': acc})
            

def test(batch_logger, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        for i, sample in enumerate(test_loader):
            sample = sample.to(device)          ##############
            end_point  = model(sample)
            loss = F.nll_loss(end_point, sample.y)
        
            pred = end_point.max(1)[1]
            acc = (pred.eq(sample.y).sum().item())/len(sample.y)
            correct += acc
        
            if i % 10 == 0:
                batch_logger.log({'batch': i + 1,'loss': loss.item(),'acc': acc})
            
        return correct / (i+1)
    
    
    
    
train_batch_logger = Logger(os.path.join('./Results', 'train_batch.log'), ['epoch', 'batch', 'loss', 'acc'])
test_batch_logger = Logger(os.path.join('./Results', 'test_batch.log'), ['batch', 'loss', 'acc'])
acc_logger = Logger(os.path.join('./Results', 'acc.log'), ['epoch', 'acc'])    
    
train_filenames = glob.glob(os.path.join('./traingraph', '*.mat'))
val_filenames = glob.glob(os.path.join('./testgraph', '*.mat'))
random.shuffle(train_filenames)
print(len(train_filenames))
print(len(val_filenames))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 150):

    train_dataset = MyOwnDataset(train_filenames)      
    test_dataset = MyOwnDataset(val_filenames)

    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True) #, collate_fn=lambda x:x)
    test_loader = DataLoader(test_dataset, batch_size=24) #, collate_fn=lambda x:x)
    
    
    
    train(epoch, train_batch_logger, train_loader)
    test_acc = test(test_batch_logger, test_loader)
    
    print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
    acc_logger.log({'epoch': epoch, 'acc': test_acc})
    
    torch.save(model, './runs_model/model.pkl')






