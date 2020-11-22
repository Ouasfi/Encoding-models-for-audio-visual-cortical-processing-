import h5py 
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict
class Linear_(torch.nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def set_weights(self, tf_weights):
      self.bias = torch.nn.Parameter(torch.from_numpy(tf_weights['b']))
      self.weight = torch.nn.Parameter(torch.from_numpy(tf_weights['W']).transpose(0,1))
      
    def forward(self, x):
      return super().forward(x)

class Permute_(nn.Module):
    def __init__(self, perm,  *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.perm = perm
    def forward(self, x):
      return x.permute(self.perm)

class Conv2d_(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def set_weights(self, tf_weights):
       self.bias = torch.nn.Parameter(torch.from_numpy(tf_weights['b']))
       self.weight = torch.nn.Parameter(torch.from_numpy(tf_weights['W']).permute(3,2,0,1))

class CommunExtractor(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()
        self.conv1 = Conv2d_(1,96, kernel_size = 9, stride = 3, padding = 3)
        self.rnorm = nn.LocalResponseNorm(  5, 1e-3, 0.75, 1)
        self.pool1 = nn.MaxPool2d(kernel_size= (3,3), stride=2)
        self.conv2 =  Conv2d_(96,256, kernel_size = 5, stride = 2, padding = 3)
        self.pool2 = nn.MaxPool2d(kernel_size= (3,3), stride = 2, padding = 1)
        self.conv3 = Conv2d_(256,512, kernel_size = 3, stride = 1, padding = 2)
        if pretrained : 
          self.load_weights() #
    def load_weights(self):

      weights_biases = np.load('./weights/network_weights_early_layers.npy', encoding = 'latin1', allow_pickle=True)[()]
      genre_branch =  np.load('./weights/network_weights_genre_branch.npy',  encoding = 'latin1', allow_pickle=True)[()]
      word_branch = np.load('./weights/network_weights_word_branch.npy',  encoding = 'latin1', allow_pickle=True)[()] 
      weights_biases.update(genre_branch)
      weights_biases.update(word_branch)
      self.tf_weights = weights_biases
      self.conv1.set_weights(self.tf_weights['conv1'])
      self.conv2.set_weights(self.tf_weights['conv2'])
      self.conv3.set_weights(self.tf_weights['conv3'])
    def forward(self, x):
        out = self.conv1(x)
        out = self.rnorm(out.relu())
        out = self.pool1(out.relu())
        out = self.conv2(out)
        out = self.rnorm(out.relu())
        out = self.pool2(out)
        out = self.conv3(out.relu())
        return out.relu()

class Branch_model(nn.Module):
    def __init__(self, branch :str, pretrained : bool = False):
        super().__init__()
        self.branch = branch
        n_classes = 587 if self.branch == "W" else 41
        self.commun_branch =  CommunExtractor(pretrained = pretrained )
        self.network = nn.Sequential(OrderedDict([
          ('conv4_'+branch, Conv2d_(512,1024, kernel_size = 3, stride = 1, padding = 2)),
          ('relu', nn.ReLU()),
          ('conv5_'+branch, Conv2d_(1024,512, kernel_size = 3, stride = 1, padding = 2)),
          ('relu', nn.ReLU()),
          ('pool3_'+branch, nn.AvgPool2d(kernel_size = 3, stride = 2 )),
          ('permute', Permute_(perm = (0,2,3,1))), #could be removed when if using tf weights
          ('flatten', nn.Flatten()),
          ('fc1_'+branch, Linear_(512*8*8,4096 )),
          ('relu', nn.ReLU()),
          ('fctop_'+branch, Linear_(4096, n_classes)),         
        ]))
        if pretrained :
          self.set_weights() 
    def set_weights(self):

      
      self.tf_weights = self.network.commun_branch.tf_weights
      getattr( self.network,'conv4_'+ self.branch).set_weights(self.tf_weights['conv4_'+ self.branch])
      getattr( self.network,'conv5_'+ self.branch).set_weights(self.tf_weights['conv5_'+ self.branch])
      getattr( self.network,'fc1_'+ self.branch).set_weights(self.tf_weights['fc6_'+ self.branch])
      getattr( self.network,'fctop_'+ self.branch).set_weights(self.tf_weights['fctop_'+ self.branch])
    def forward(self, x):
        x = self.commun_branch(x)
        return self.network(x)
