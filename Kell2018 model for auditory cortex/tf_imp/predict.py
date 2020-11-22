import tensorflow as tf
import numpy as np
from models import *

def get_label(model, c_gram, type = None ):

  keys = music_key if type == "music" else word_key
  logits =model(c_gram.astype(float32), type = type)
  prediction = logits.numpy().argsort()[:,-5:][0][::-1]
  labels = list(map(lambda s : s.decode("utf-8"), keys[prediction]))
  print ("Predicted labels :" + "\n"+ "; ".join(list(map(lambda s : s.decode("utf-8"), keys[prediction]))))

  return labels

if __name__ == "__main__":

    branched_model = branched_network() # make network object
    word_key = np.load('../demo_stim/logits_to_word_key.npy', allow_pickle = True,) #Load logits to word key 
    music_key = np.load('../demo_stim/logits_to_genre_key.npy', allow_pickle = True,) #Load logits to genre key

    example_cochleagram = np.load('../demo_stim/example_cochleagram_0.npy', allow_pickle = True,) 
    labels = get_label(branched_model, example_cochleagram )
    print("Speech Example ... actual label: according ")

