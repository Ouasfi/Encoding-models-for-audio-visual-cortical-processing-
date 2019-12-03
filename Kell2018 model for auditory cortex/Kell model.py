import h5py 
import numpy as np
import os
import torch
import torch.nn.functional as F





class Word_Music(object):
  
  def __init__(self, input, weights_biases):
    super(Word_Music,self).__init__()
    
    self.rnorm_bias, self.rnorm_alpha , self.rnorm_beta = 1., 1e-3, 0.75
    self.n_out_pool5_W = 6 * 6 * 512
    self.input = input
    self.weights_biases = weights_biases
    self.summary = {}
    
    self.logits = self.forward()
    
  def load(self, path):
    self.weights_biases = np.load(os.getcwd + path)[()]
    return self
  
  def torch_conv2d( self, x_torch, conv_name, padding, stride):
    if torch.cuda.is_available:
      device = torch.device("cuda")
      x_torch = x_torch.to(device)
      weights_torch = torch.Tensor(self.weights_biases[conv_name]['W']).permute((3, 2, 0, 1)).to(device)
      biases_torch = torch.Tensor(self.weights_biases[conv_name]['b']).to(device)
      return F.relu(F.conv2d(x_torch, weights_torch,biases_torch, stride, padding))
  
  def torch_pool(self, input, kernel_size, stride=None, padding=0, max_pool = True):
    if max_pool:
      return F.max_pool2d(input, kernel_size, stride=None, padding=0)
    elif not max_pool:
      return F.avg_pool2d(input, kernel_size, stride=None, padding=0)
  def lr_norm (self, input, size):
    return F.local_response_norm(input, size, self.rnorm_alpha, self.rnorm_beta, self.rnorm_bias)
  
  def reshape_layer(self, previous_layer, output_size):
    
    previous_layer = torch.from_numpy(previous_layer).type(torch.float32)
    return previous_layer.view(-1, output_size[0],output_size[1],output_size[2])
  def linear_layer (self, input, layer_name):
    
    size = 1
    for k in range(len(input.size())):
      size = input.size()[k]*size
      
    
    
    input = input.view(-1, size)
    if torch.cuda.is_available:
      device = torch.device("cuda")
      weights_torch = torch.Tensor(self.weights_biases[layer_name]['W']).to(device)
      biases_torch = torch.Tensor(self.weights_biases[layer_name]['b']).to(device)
      return  F.relu(torch.matmul(input,weights_torch)+ biases_torch)
      
    
  
    
  
  def forward (self):
    
    x = self.reshape_layer(self.input,(256,256,1))
    summary['initial size'] =  x.size()
    x = x.permute((3,0,2,1))
    x = self.torch_conv2d(x, conv_name = 'conv1', padding=1, stride= 2)
    summary['conv1'] =x.size()
    x = self.lr_norm(x, size= 2 )
    summary['lr norm'] = x.size()
    x = self.torch_pool( x,kernel_size= (3, 3), stride = (2,2), padding = 1)#.permute(0,3,2,1)
    summary['max pool'] =  x.size()
    x = self.torch_conv2d(x, 'conv2', padding=1, stride=2)
    summary['conv2']=x.size()
    x = self.lr_norm( x, size = 2)
    summary['lrnorm'] =  x.size()
    x = self.torch_pool( x,kernel_size= (3, 3), stride = (2,2), padding = 1)
    summary['max pool'] =  x.size()
    x = self.torch_conv2d(x, 'conv3', padding= 1, stride = 1)
    summary['conv3']=x.size()
    
    #speech branch
    x = self.torch_conv2d(x, conv_name = 'conv4_W', padding = 1, stride = 1 )
    summary['conv4_W']=x.size()
    x = self.torch_conv2d(x, conv_name = 'conv5_W', padding = 1, stride = 1 )
    summary['conv5_W']=x.size()
    x = self.torch_pool( x,kernel_size= (1, 1), stride = (2,2), padding = 1,max_pool = False )
    summary['max pool'] =  x.size()
    x = self.linear_layer(x,'fc6_W')
    summary['fc6_W'] = x.size()
    x = self.linear_layer(x,'fctop_W')
    summary['fctop_w'] =  x.size()
    return x