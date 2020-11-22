from pycochleagram.pycochleagram import cochleagram as cgram 
from PIL import Image
import Kell model
import IPython.display as ipd
import sys
sys.path.append('./network/')
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
    print (title+':')
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





def logits_predict(input_data, actual_label, model,  wav= False, music = False):
	"""
	input_data : (sr, wav_f )
	actual_label : the word corresponding to the audio
	model : a kell model to compute  logits with the nn.functionnal modules of pytorch. Weights should be associated to the model before assignement.
	"""
	if wav:
		sr, wav_f = input_data
		play_wav(wav_f, sr, actual_label)
		input_data  = generate_cochleagram(wav_f, sr, actual_label)

	net_object =model(input_data) # make network object
	word_key = np.load('../demo_stim/logits_to_word_key.npy')#Load logits to word key
	music_key = np.load('../demo_stim/logits_to_genre_key.npy')#Load logits to genre key
	logits = net_object.logits






