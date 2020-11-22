
from pycochleagram import cochleagram as cgram
import h5py 
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import IPython.display as ipd
import sys

import tensorflow as tf
import scipy.io.wavfile as wav
import matplotlib as plt 

def resample(example, new_size):
    im = Image.fromarray(example)
    resized_image = im.resize(new_size, resample=Image.ANTIALIAS)
    return np.array(resized_image)

def plot_cochleagram(cochleagram, title): 
    plt.figure(figsize=(6,3))
    plt.matshow(cochleagram.reshape(256,256), origin='lower',cmap=plt.cm.Blues, fignum=False, aspect='auto')
    plt.yticks([]); plt.xticks([]); plt.title(title); 
    
def play_wav(wav_f, sr, title):   
    print( title+':')
    ipd.display(ipd.Audio(wav_f, rate=sr))

def generate_cochleagram(wav_f, sr, title):
    # define parameters
    n, sampling_rate = 50, 16000
    low_lim, hi_lim = 20, 8000
    sample_factor, pad_factor, downsample = 4, 2, 200
    nonlinearity, fft_mode, ret_mode = 'power', 'auto', 'envs'
    strict = True

    # create cochleagram
    c_gram = cgram.cochleagram(wav_f, sr, n, low_lim, hi_lim, 
                               sample_factor, pad_factor, downsample,
                               nonlinearity, fft_mode, ret_mode, strict)
    
    # rescale to [0,255]
    c_gram_rescaled =  255*(1-((np.max(c_gram)-c_gram)/np.ptp(c_gram)))
    
    # reshape to (256,256)
    c_gram_reshape_1 = np.reshape(c_gram_rescaled, (211,400))
    c_gram_reshape_2 = resample(c_gram_reshape_1,(256,256))
    
    plot_cochleagram(c_gram_reshape_2, title)

    # prepare to run through network -- i.e., flatten it
    c_gram_flatten = np.reshape(c_gram_reshape_2, (1, 256*256)) 
    
    return c_gram_flatten