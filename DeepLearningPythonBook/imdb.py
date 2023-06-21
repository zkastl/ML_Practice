import numpy as np
import os
import tensorflow as tf

from keras import utils as kutils
from keras.layers import Dense, Embedding, Flatten
from keras.models import Sequential


maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000
log_dir = '.'
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
imdb_dir = '../data/aclImdb'
train_dri = os.path.join(imdb_dir, 'train')

labels = []
texts = []