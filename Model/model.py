# import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import MeanSquaredError
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime

NAME = "test-{}".format(datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))

print("Using tensorflow " + tf.__version__)
tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))

# load training data
samples_json = open("train_data/samples_2020-5-18--16-25-28.json", "r")
samples = json.load(samples_json)
# print(samples)

# bring training data into right shape
print("preprocessing input training data...")
x_train = np.array([np.array(data['x_train']) for data in samples])
# print(x_train)

print("preprocessing output training data...")
y_train = np.array([np.array([data['y_train']]) for data in samples])
# print(y_train)

print("creating model...")
model = tf.keras.models.Sequential();

model.add(Dense(64, input_shape=(23,)))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('relu'))

print(model.summary())

#
print("compiling model...")
model.compile(
    optimizer='adam',
    # loss='mean_squared_error',
    # loss='mean_absolute_error',
    loss='cosine_similarity',
    metrics=['accuracy'],
)

#
print("training model...")
model.fit(x_train, y_train, epochs=500, callbacks=[tensorboard])

model.save("pt1.model")

# # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, dpi=96*2)