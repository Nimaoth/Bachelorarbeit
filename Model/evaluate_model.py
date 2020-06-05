# import tensorflow.keras as keras
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import MeanSquaredError
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime

NAME = "test-2020-06-05--16-55-01"

# load training data
samples_json = open("train_data/samples_2020-5-18--16-25-28.json", "r")
samples = json.load(samples_json)

# bring training data into right shape
print("preprocessing input training data...")
x_train1 = np.array([np.array(data['x_train']) for data in samples])
x_train2 = np.array([np.array(data['y_train']) for data in samples])
# print(x_train1)
# print(x_train2)

print("preprocessing output training data...")
y_train1 = np.array([np.array([data['y_train']]) for data in samples])
y_train2 = np.array([0.5 for data in samples])
# print(y_train1)
# print(y_train2)

vae = K.models.load_model("models/" + NAME)

for i in range(0, 100):
    result = vae.predict([x_train1[i:i+1], x_train2[i:i+1]])
    print("expected: ", end="")
    print(y_train1[i], end="")
    print(", ", end="")
    print(y_train2[i], end="")
    print(", got: ", end="")
    print(result)

print("===============")
