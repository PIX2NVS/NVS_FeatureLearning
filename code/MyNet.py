from __future__ import division
import os.path as osp
import numpy as np
import math
import shutil
import glob
import os
import sys
from collections import Counter
from utils import Logger

import torch
from torch.utils.data.dataset import Subset
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
import torch.nn.functional as F
from inputsdata import MyOwnDataset
import torch_geometric.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch_geometric.utils import normalized_cut
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x, graclus, global_mean_pool, NNConv


from torch.autograd import Variable
from collections import OrderedDict


NUM = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left_conv1 = SplineConv(in_channel, out_channel, dim=2, kernel_size=4)
        self.left_bn1 = torch.nn.BatchNorm1d(out_channel)
        
        self.shortcut_conv = SplineConv(in_channel, out_channel, dim=2, kernel_size=1)
        self.shortcut_bn = torch.nn.BatchNorm1d(out_channel)
             
    def forward(self, data):
        
        data.x = F.elu(self.left_bn1(self.left_conv1(data.x, data.edge_index, data.edge_attr)) + 
                       self.shortcut_bn(self.shortcut_conv(data.x, data.edge_index, data.edge_attr)))
        
        return data

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ResidualBlock(2, 32)
        self.conv2 = ResidualBlock(32, 64)
        self.conv3 = ResidualBlock(64, 128)
        
        self.conv3d1 = Unit3D(in_channels=128, output_channels=256, kernel_shape=[3, 3, 3], name='3d1')   # [8,30,30,256]  
        self.pool1 = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(1, 2, 2), padding=0)    # [8,15,15,256]                    
        self.conv3d2 = Unit3D(in_channels=256, output_channels=512, kernel_shape=[3, 3, 3], name='3d2')   
        self.pool2 = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)    # [4,8,8,512]
        self.conv3d3 = Unit3D(in_channels=512, output_channels=512, kernel_shape=[3, 3, 3], name='3d3')   
        self.pool3 = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)    # [2,4,4,512]
        self.conv3d4 = Unit3D(in_channels=512, output_channels=1024, kernel_shape=[3, 3, 3], name='3d4')   
        self.pool4 = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2), padding=0)    # [2,2,2,1024]
      
        
        self.avg_pool = nn.AvgPool3d(kernel_size=[1, 2, 2], stride=(1, 2, 2))
        self.logits = Unit3D(in_channels=1024, output_channels=51,
                             kernel_shape=[1, 1, 1],
                             padding=0,
                             activation_fn=None,
                             use_batch_norm=False,
                             use_bias=True,
                             name='logits')
        


    def forward(self, data):

        #print(data.node_num)
        #print(data.batch)
        #print(data.batch[-1]+1)
  
        
        for i in range(NUM*(data.batch[-1]+1)):        ##########################  NUM * BatchSize
            if i == 0:
                #print(data.node_num[i])
                batch = np.ones(data.node_num[i])
            else:
                #print(data.node_num[i])
                batch = np.append(batch, np.ones(data.node_num[i])*i)
        batch = torch.tensor(batch, dtype=torch.int64)
        data.batch = batch.cuda()
        
        data = self.conv1(data)
        cluster = voxel_grid(data.pos, data.batch, size=[2,2])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.conv2(data)
        cluster = voxel_grid(data.pos, data.batch, size=[4,3])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
        data = self.conv3(data)
        cluster = voxel_grid(data.pos, data.batch, size=[8,6])
        x = max_pool_x(cluster, data.x, data.batch, size=900)
        
        
        x = torch.reshape(x, (-1, NUM,30,30,128))     #(B,T,H,W,C)             ##############################
        #print(x.shape)
        x = x.permute([0,4,1,2,3])              #(B,C,T,H,W)
        
        x = self.conv3d1(x)    
        x = self.pool1(x)        
        x = self.conv3d2(x)
        x = self.pool2(x)
        x = self.conv3d3(x)
        x = self.pool3(x)
        x = self.conv3d4(x)
        x = self.pool4(x)
        
        x = self.avg_pool(x)
        x = F.dropout(x, training=self.training)
        x = self.logits(x)
        x = torch.squeeze(x)
        

        return F.log_softmax(x, dim=1)




class MaxPool3dSamePadding(nn.MaxPool3d):
    
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self.stride[0]))
        out_h = np.ceil(float(h) / float(self.stride[1]))
        out_w = np.ceil(float(w) / float(self.stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)
    

class Unit3D(nn.Module):

    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        
        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0, # padding to be 0 here. We will dynamically pad based on input size in forward function
                                bias=self._use_bias)
        
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

            
    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        #print t,h,w
        out_t = np.ceil(float(t) / float(self._stride[0]))
        out_h = np.ceil(float(h) / float(self._stride[1]))
        out_w = np.ceil(float(w) / float(self._stride[2]))
        #print out_t, out_h, out_w
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)
        #print pad_t, pad_h, pad_w

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)
        #print x.size()
        #print pad
        x = F.pad(x, pad)
        #print x.size()        

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x



class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0], kernel_shape=[1, 1, 1], padding=0,
                         name=name+'/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4], kernel_shape=[3, 3, 3],
                          name=name+'/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5], kernel_shape=[1, 1, 1], padding=0,
                          name=name+'/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):    
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0,b1,b2,b3], dim=1)



