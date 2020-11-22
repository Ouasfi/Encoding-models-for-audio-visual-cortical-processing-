from models import *
import tensorflow as tf

def branch_model(branch_name : str):
    branched_model = branched_network()
    commun = branched_model.commun
    branch = branched_model.music if branch_name == "music" else branched_model.speech
    return tf.keras.Sequential(commun + branch )
def train_( model_, optimizer : str, loss : str,  batch_size : int , x , y):
    model_.compile(optimizer = optimizer, loss = loss)
    model_.fit(x, y, batch_size = batch_size)
    return model_

if __name__ == "__main__":
    #simulated data
    x = tf.random.normal((500,65536))
    w = tf.random.normal(( 65536, 43))
    y = tf.nn.sigmoid(x@w) ; y.shape
    #model
    music_model = branch_model(branch_name = "music")
    print(music_model.summary())
    # train
    train_(music_model,optimizer =  "adam", loss = "mse", batch_size = 30, x, y)

