import numpy as np
import os
import tensorflow as tf

from keras import utils as kutils
from keras.layers import Dense, Embedding, Flatten
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer


maxlen = 100
training_samples = 200
validation_samples = 10000
max_words = 10000
log_dir = '.'
tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
imdb_dir = '../data/aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            with open(os.path.join(dir_name, fname)) as f:
                texts.append(f.read())

            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)