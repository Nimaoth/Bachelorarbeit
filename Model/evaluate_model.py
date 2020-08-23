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

NAME = "test-2020-07-04--18-10-52"

arr = np.array([1, 2, 3])

#load model
vae = K.models.load_model("models/" + NAME)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_weights(layers, name):
    json_dump = json.dumps(layers.get_weights(), cls=NumpyEncoder)
    with open("weights/" + NAME + "_" + name + ".json", "w") as file:
        file.write(json_dump)

save_weights(vae.features, "features")
save_weights(vae.absorbtion, "absorbtion")
save_weights(vae.decoder, "decoder")

# # load training data
# samples_json = open("../train_data/samples_2020-6-9--15-31-56.json", "r")
# samples = json.load(samples_json)


# # bring training data into right shape
# print("preprocessing input training data...")
# x_train_features = np.array([np.array(data['features']) for data in samples])
# x_train_out_pos = np.array([np.array(data['out_pos']) for data in samples])
# # print(x_train_features)
# # print(x_train_out_pos)

# print("preprocessing output training data...")
# y_train_out_pos = np.array([np.array([data['out_pos']]) for data in samples])
# y_train_absorbtion = np.array([np.array([data['absorbtion']]) for data in samples])
# # print(y_train_out_pos)
# # print(y_train_absorbtion)

# print(vae.decoder.get_weights())

# # for i in range(0, 100):
# #     result = vae.predict([x_train_features[i:i+1], x_train_out_pos[i:i+1]])
# #     print("expected: ", end="")
# #     print(y_train_out_pos[i], end="")
# #     print(", ", end="")
# #     print(y_train_absorbtion[i], end="")
# #     print(", got: ", end="")
# #     print(result)

# # print("===============")
