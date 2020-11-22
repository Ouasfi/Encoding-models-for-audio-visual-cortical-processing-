import tensorflow as tf
import numpy as np
class  conv_layer(tf.keras.layers.Layer):
  def __init__(self,layer_params, layer_vars,  *args, **kwargs):
    super( conv_layer, self).__init__(*args, **kwargs)
    self.layer_params = layer_params


  #def build(self, layer_vars):
    self.w = tf.Variable(layer_vars['W'], trainable=True)
    self.b = tf.Variable(layer_vars['b'], trainable= True)
  # Call method will sometimes get used in graph mode,
  # training will get turned into a tensor
  @tf.function
  def call(self, previous_h):
    
    x = tf.nn.conv2d( previous_h, self.w, 
                                  strides=[1,self.layer_params['stride'], self.layer_params['stride'],1],
                              padding='SAME' ) + self.b
    x = tf.nn.relu(x)
    return x 

class  lrnorm_layer(tf.keras.layers.Layer):
  def __init__(self,layer_params,rnorm_bias ,rnorm_alpha , rnorm_beta, **kwargs ):
    super( lrnorm_layer, self).__init__( **kwargs)
    self.depth_radius = layer_params['radius']
    self.rnorm_bias = rnorm_bias
    self.rnorm_alpha = rnorm_alpha
    self.rnorm_beta = rnorm_beta
  
  @tf.function
  def call(self, previous_h):
    
    x = tf.nn.local_response_normalization(previous_h,
                                                     depth_radius = self.depth_radius,
                                                     bias = self.rnorm_bias,
                                                     alpha = self.rnorm_alpha,
                                                     beta = self.rnorm_beta )
    
    return x 

class  pool_layer(tf.keras.layers.Layer):
  def __init__(self,layer_params, max_pool = True,**kwargs ):
    super( pool_layer, self).__init__(**kwargs)
    self.layer_params = layer_params
    self.max_pool = max_pool
  
  @tf.function
  def call(self, previous_h):
    
    if self.max_pool: 
            return tf.nn.max_pool(  previous_h,
                                    ksize=[1,int(self.layer_params['edge']),int(self.layer_params['edge']),1],
                                    strides=[1,int(self.layer_params['stride']),int(self.layer_params['stride']),1],
                                    padding='SAME' )
        
                            
    elif not self.max_pool:
        return tf.nn.avg_pool(  previous_h, 
                                ksize=[1, self.layer_params['edge'],self.layer_params['edge'],1],
                                strides=[1, self.layer_params['stride'], self.layer_params['stride'],1],
                                padding='SAME' )
    return x 

class  flatten_pool_layer(tf.keras.layers.Layer):
  def __init__(self,output_size, *args,**kwargs ):
    super( flatten_pool_layer, self).__init__(*args, **kwargs)
    
    self.output_size = output_size
  
  @tf.function
  def call(self, previous_h):
    x = tf.reshape(previous_h, [-1, self.output_size])
    return x 

class  linear_pool_layer(tf.keras.layers.Layer):
  def __init__(self,layer_vars_dict, branch, *args,**kwargs ):
    super( linear_pool_layer, self).__init__(*args, **kwargs)
    
    self.fc6_W = tf.Variable(layer_vars_dict['fc6_'+ branch]['W'], trainable=True)
    self.fc6_b = tf.Variable(layer_vars_dict['fc6_' + branch ]['b'], trainable= True)
    self.fctop_W = tf.Variable(layer_vars_dict['fctop_' + branch]['W'], trainable=True)
    self.fctop_b = tf.Variable(layer_vars_dict['fctop_' + branch]['b'], trainable=True)

  
  @tf.function
  def call(self, previous_h):
    x = tf.matmul(previous_h, self.fc6_W) + self.fc6_b 
    x = tf.nn.relu(x)
    x = tf.matmul(x,  self.fctop_W) + self.fctop_b

    return x

class reshape_layer(tf.keras.layers.Layer):
  def __init__(self,layer_params, *args,**kwargs ):
    super( reshape_layer, self).__init__(*args, **kwargs)
    
    self.size = layer_params['edge']
  
  @tf.function
  def call(self, previous_h):
    x= tf.reshape(previous_h, [-1,self.size, self.size,1])
    return x 


rnorm_bias, rnorm_alpha, rnorm_beta = 1., 1e-3, 0.75
lrnorm = lambda s : lrnorm_layer(s, rnorm_bias = rnorm_bias, rnorm_alpha = rnorm_alpha, rnorm_beta =rnorm_beta )
n_out_pool5_W = 6 * 6 * 512 
n_out_pool5_G = 6 * 6 * 512

class branched_network(tf.keras.Model):

  def __init__(self, weights = None):
    super(branched_network, self).__init__()

    self.layer_params_dict = {
                'data':{'edge': 256},
                'conv1': {'edge': 9, 'stride': 3, 'n_filters': 96},
                'rnorm1': {'radius': 2}, 
                'pool1': {'edge': 3, 'stride': 2},
                'conv2': {'edge': 5, 'stride': 2, 'n_filters': 256},
                'rnorm2': {'radius': 2}, 
                'pool2': {'edge': 3, 'stride': 2},
                'conv3': {'edge': 3, 'stride': 1, 'n_filters': 512},
                'conv4_W': {'edge': 3, 'stride': 1, 'n_filters': 1024},
                'conv4_G': {'edge': 3, 'stride': 1, 'n_filters': 1024},
                'conv5_W': {'edge': 3, 'stride': 1, 'n_filters': 512},
                'conv5_G': {'edge': 3, 'stride': 1, 'n_filters': 512},
                'pool5_W': {'edge': 3, 'stride': 2},
                'pool5_G': {'edge': 3, 'stride': 2},
                'fc6_W': {'n_units': 1024},
                'fc6_G': {'n_units': 1024},
                'fctop_W': {'n_units': 589},
                'fctop_G': {'n_units': 43}
        }
    self.load_weights() if weights is None else self.set_weights(weights)
    
    self.commun =  [
        tf.keras.layers.InputLayer(input_shape=(self.layer_params_dict['data']['edge']*self.layer_params_dict['data']['edge'])),
        reshape_layer( self.layer_params_dict['data']),                                
        conv_layer( self.layer_params_dict['conv1'], self.layer_vars_dict['conv1']),
        lrnorm(self.layer_params_dict['rnorm1']),
        pool_layer( self.layer_params_dict['pool1']),
        conv_layer(self.layer_params_dict['conv2'], self.layer_vars_dict['conv2']),
        lrnorm( self.layer_params_dict['rnorm2']),
        pool_layer( self.layer_params_dict['pool2']),
        conv_layer(self.layer_params_dict['conv3'], self.layer_vars_dict['conv3'])
      ]
    self.speech  = [ 
      conv_layer(self.layer_params_dict['conv4_W'], self.layer_vars_dict['conv4_W']),
      conv_layer(self.layer_params_dict['conv5_W'],  self.layer_vars_dict['conv5_W']),
      pool_layer( self.layer_params_dict['pool5_W'],  False),
      flatten_pool_layer(  output_size = n_out_pool5_W ),
      linear_pool_layer(self.layer_vars_dict, branch = "W")
        
      ]
    self.music  = [ 
      conv_layer(self.layer_params_dict['conv4_G'], self.layer_vars_dict['conv4_G']),
      conv_layer(self.layer_params_dict['conv5_G'],  self.layer_vars_dict['conv5_G']),
      pool_layer( self.layer_params_dict['pool5_G'],  False),
      flatten_pool_layer(  output_size = n_out_pool5_G),
      linear_pool_layer(self.layer_vars_dict, branch = "G")
    ]

  def load_weights(self):

    weights_biases = np.load('./weights/network_weights_early_layers.npy', encoding = 'latin1', allow_pickle=True)[()]
    genre_branch =  np.load('./weights/network_weights_genre_branch.npy',  encoding = 'latin1', allow_pickle=True)[()]
    word_branch = np.load('./weights/network_weights_word_branch.npy',  encoding = 'latin1', allow_pickle=True)[()] 
    weights_biases.update(genre_branch)
    weights_biases.update(word_branch)
    self.layer_vars_dict = weights_biases

  def set_weights(self, weights):
    self.layer_vars_dict = weights

  def call(self, x, type):

    for layer in self.commun : x = layer(x)
    if type == "speech":
      for layer in self.speech: x = layer(x)
    if type == "music":
      for layer in self.music: x = layer(x)

    return x