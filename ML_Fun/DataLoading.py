import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

np.set_printoptions(precision=3, suppress=True)

abalone_train = pd.read_csv(
    "https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])

abalone_train.head()