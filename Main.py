# (v1 uses tensorflow 1, but v2 will use pyTorch)
# as this is v1.0, the code is not optimized at maximum, and in the data preprocessing part, is not efficient
# this program obviously uses my own paths, if you copy my code make sure to adapt them


# Data preprocessing of images

import numpy as np
from PIL import Image, ImageOps
import numpy as np
from os import listdir
from os.path import isfile, join

"C:\\Users\\User\\Pictures\\{}\\images"
Xarr = []
folders = ['KinFaceW-I','KinFaceW-II']
maxsize = []

for folder in folders:
    for f in listdir("C:\\Users\\User\\Pictures\\{}\\images".format(folder)):
        if isfile(join("C:\\Users\\User\\Pictures\\{}\\images".format(folder), f)):
            imgpil = Image.open("C:\\Users\\User\\Pictures\\{}\\images\\{}".format(folder,f))
            maxsize.append(np.array(imgpil).shape[0])
maxsize = max(maxsize)
   
for folder in folders:    
    for f in listdir("C:\\Users\\User\\Pictures\\{}\\images".format(folder)):
        if isfile(join("C:\\Users\\User\\Pictures\\{}\\images".format(folder), f)):
            imgpil = Image.open("C:\\Users\\User\\Pictures\\{}\\images\\{}".format(folder,f))
            imgarr = np.array(ImageOps.expand(imgpil,int((maxsize-imgpil.size[0])/2),fill='black'))
            imgarr = imgarr.reshape(-1)
            if np.any(np.isnan(imgarr.reshape(-1))):
                print("nan")
            elif imgarr.shape == (maxsize*maxsize*3,):
                Xarr.append(imgarr/255)
            else:
                print(imgarr.shape)
Xarr = np.array(Xarr)
print(len(Xarr))

def next_batch(num, data): # Defining the next batch function, wich will provide the batch for every iteration
    id = np.arange(0 , len(data)) # Storing all possible index of the dataset
    np.random.shuffle(idx) # Shuffle of all indexs
    id = id[:num] # Keeping only the number of indexes wanted
    data_shuffle = np.array([data[i] for i in id]) Storing all datas corresponding to the images at the index i
    return data_shuffle


# Conception du schema tensorflow
import tensorflow as tf
from functools import partial

n_inputs = maxsize**2*3 # couche d'entrée
n_hidden1 = 10000
n_hidden2 = 8192
n_hidden2p1 = 4096
n_hidden3 = 2048 # couche de codages
n_hidden3p1 = n_hidden2p1
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs # sortie
learning_rate = 0.001

initializer = tf.contrib.layers.variance_scaling_initializer()
my_dense_layer = partial(
tf.layers.dense,
activation = tf.nn.elu,
kernel_initializer=initializer)

X = tf.placeholder(tf.float32,[None,n_inputs])
hidden1 = my_dense_layer(X,n_hidden1)
hidden2 = my_dense_layer(hidden1,n_hidden2)
hidden2p1 = my_dense_layer(hidden2,n_hidden2p1)
hidden3_mean = my_dense_layer(hidden2p1,n_hidden3,activation=None)
hidden3_gamma = my_dense_layer(hidden2p1,n_hidden3,activation=None)
noise = tf.random_normal(tf.shape(hidden3_gamma),dtype=tf.float32)
hidden3 = hidden3_mean + tf.exp(0.5*hidden3_gamma) * noise
hidden3p1 = my_dense_layer(hidden3,n_hidden3p1)
hidden4 = my_dense_layer(hidden3p1,n_hidden4)
hidden5 = my_dense_layer(hidden4,n_hidden5)
logits = my_dense_layer(hidden5,n_outputs,activation=None)
outputs = tf.nn.sigmoid(logits) # as we use sigmoid, all the values of the pixels must be 0=<value=<1, so in the preprocessing part, we divided by 255 each values

entropie = tf.nn.sigmoid_cross_entropy_with_logits(labels=X,logits=logits)
perte_r = tf.reduce_sum(entropie)
perte_l = 0.5*tf.reduce_sum(tf.exp(hidden3_gamma) + tf.square(hidden3_mean)-1-hidden3_gamma)
loss = perte_r + perte_l

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
#optimizer.minimize(loss)
init = tf.global_variables_initializer()

# Entrainement de l'auto-encodeur
nb_gen = 60
nepochs = 200 
batch_size = 150

with tf.Session() as sess:
    init.run() # weights initialisation (here, He initialisation)
    for ep in range(nepochs): # epochs
        n_batches = Xarr.shape[0] // batch_size
        for iteration in range(n_batches): # iterations
            X_batch = next_batch(batch_size,Xarr)
            sess.run(optimizer.minimize(loss), feed_dict={X: X_batch})
    
    noise_bis = np.random.normal(size=[n_digits,n_hidden3]) # Noise creation
    generations = outputs.eval(feed_dict={hidden3: codings_rnd}) # Transformation of the noise into images, by processing the noise throught
    layers (>3) of the network.
