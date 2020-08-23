# import tensorflow.keras as keras
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import json
from datetime import datetime
from pathlib import Path

NAME = "test-{}".format(datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
TRAIN_DATA = "samples_fixed_normal_100_20000"
EPOCHS1 = 20
EPOCHS2 = 5
LEARNING_RATE = 0.0002

USE_FIXED_NORMAL_DIST = False
FIXED_NORMAL_DIST_VAR = 20.0

# KL_LOSS_WEIGHT_GRAPH = [(25, 0.0, 0.0)]
KL_LOSS_WEIGHT_GRAPH = [(20, 0.0, 1.0), (5, 1.0, 1.0)]

# intermediate_act = "relu"
intermediate_act = LeakyReLU(alpha=0.1)

intermediate_dim = 64
absorbtion_int_dim = 32
absorbtion_output_dim = 1

encoder_input_shape = (67, )
decoder_input_shape = (68, )
feature_input_shape = (23, )
feature_output_shape = (64, )
absorbtion_input_shape = (64, )
latent_dim = 4
decoder_output_dim = 3
latent_shape = (2, )

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



print("Using tensorflow " + tf.__version__)
tensorboard = TensorBoard(log_dir="logs2\\{}".format(NAME))

# load training data
print("opening file")
samples_json = open("../train_data/" + TRAIN_DATA + ".json", "r")
print("loading json")
samples = json.load(samples_json)
samples_json.close()
# print(samples)

# bring training data into right shape
print("preprocessing input training data...")
x_train_features = np.array([np.array(data['features']) for data in samples])
x_train_out_pos = np.array([np.array(data['out_pos']) for data in samples])

print("preprocessing output training data...")
y_train_out_pos = x_train_out_pos
y_train_absorbtion = np.array([np.array([data['absorbtion']]) for data in samples])

samples = None

# shuffle training data
def unison_shuffled_copies(a, b, c, d):
    assert len(a) == len(b) and len(a) == len(c) and len(a) == len(d)
    p = np.random.permutation(len(a))
    return a[p], b[p], c[p], d[p]

x_train_features, x_train_out_pos, y_train_out_pos, y_train_absorbtion = unison_shuffled_copies(
    x_train_features, x_train_out_pos, y_train_out_pos, y_train_absorbtion
    )

print("sample count: " + str(len(x_train_features)))

assert not np.any(np.isnan(x_train_features))
assert not np.any(np.isnan(x_train_out_pos))
assert not np.any(np.isnan(y_train_out_pos))
assert not np.any(np.isnan(y_train_absorbtion))

print("creating model...")

class Encoder(layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(name="Encoder", **kwargs)
        self.dense1 = Dense(intermediate_dim, activation=intermediate_act)
        self.dense2 = Dense(intermediate_dim, activation=intermediate_act)
        self.dense3 = Dense(intermediate_dim, activation=intermediate_act)
        self.mean = Dense(latent_dim)
        self.log_var = Dense(latent_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)

        z_mean = self.mean(x)
        z_log_var = self.log_var(x)

        if USE_FIXED_NORMAL_DIST:
            z = K.backend.random_normal(shape=(tf.shape(z_mean)[0], 4), mean=0.0, stddev=FIXED_NORMAL_DIST_VAR)
        else:
            z = z_mean + tf.exp(0.5 * z_log_var) * K.backend.random_normal(shape=(tf.shape(z_mean)[0], 4))

        return z_mean, z_log_var, z


class Decoder(layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(name="Decoder", **kwargs)
        self.dense1 = Dense(intermediate_dim, activation=intermediate_act, input_shape=decoder_input_shape)
        self.dense2 = Dense(intermediate_dim, activation=intermediate_act)
        self.dense3 = Dense(intermediate_dim, activation=intermediate_act)
        self.dense_output = Dense(decoder_output_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)

        x = self.dense_output(x)
        return x

class Features(layers.Layer):
    def __init__(self, **kwargs):
        super(Features, self).__init__(name="Features", **kwargs)
        self.dense1 = Dense(intermediate_dim, activation=intermediate_act, input_shape=feature_input_shape)
        self.dense2 = Dense(intermediate_dim, activation=intermediate_act)
        self.dense3 = Dense(intermediate_dim, activation=intermediate_act)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

class Absorbtion(layers.Layer):
    def __init__(self, **kwargs):
        super(Absorbtion, self).__init__(name="Absorbtion", **kwargs)
        self.dense1 = Dense(absorbtion_int_dim, activation=intermediate_act, input_shape=absorbtion_input_shape)
        self.dense2 = Dense(absorbtion_output_dim, activation="sigmoid")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

class VariationalAutoEncoder(tf.keras.Model):
    def __init__(self, **kwargs):
        super(VariationalAutoEncoder, self).__init__(name="Vae", **kwargs)
        self.features = Features()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.absorbtion = Absorbtion()
        self.kl_loss_weight = 1.0

    def call(self, inputs):
        feature_inputs, encoder_inputs = inputs
        latent_features = self.features(feature_inputs)
        # z_mean, z_log_var, z = self.encoder(layers.concatenate([latent_features, encoder_inputs]))
        z_mean, z_log_var, z = self.encoder(encoder_inputs)

        reconstructed = self.decoder(layers.concatenate([latent_features, z]))
        absorbtion_result = self.absorbtion(latent_features)

        z_var = tf.exp(z_log_var)
        kl_loss = 0.5 * (tf.reduce_sum(tf.exp(z_log_var)) + tf.reduce_sum(tf.square(z_mean)) - 4 - tf.reduce_sum(z_log_var))
        kl_loss *= tf.constant(self.kl_loss_weight)

        self.add_loss(kl_loss)
        self.add_metric(kl_loss, aggregation="mean", name="kl_loss")
        self.add_metric(tf.constant(self.kl_loss_weight), aggregation="mean", name="kl_loss_weight")

        return reconstructed, absorbtion_result

class CustomCallback(K.callbacks.Callback):
    def __init__(self, vae, **kwargs):
        super(CustomCallback, self).__init__(**kwargs)
        self.vae = vae

    def on_train_begin(self, logs=None):
        self.save_weights("init")

    def on_train_end(self, logs=None):
        self.save_weights("final")

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0:
            vae.save("models/" + NAME)
        # if epoch % 5 == 0:
        self.save_weights(str(epoch))

    def save_weights(self, step):
        def save_weights_impl(layers, name):
            json_dump = json.dumps(layers.get_weights(), cls=NumpyEncoder)
            # with open("weights/" + NAME + "_" + name + ".json", "w") as file:
            Path("weights/{}/{}".format(NAME, step)).mkdir(parents=True, exist_ok=True)
            with open("weights/{}/{}/{}.json".format(NAME, step, name), "w") as file:
                file.write(json_dump)
            self.vae.save_weights("weights/{}/{}/all".format(NAME, step))

        save_weights_impl(self.vae.features, "features")
        save_weights_impl(self.vae.absorbtion, "absorbtion")
        save_weights_impl(self.vae.decoder, "decoder")

vae = VariationalAutoEncoder()
vae.save_weights("temp_weights")


current_epoch = 0
for i in range(0, len(KL_LOSS_WEIGHT_GRAPH)):
    epochs, weight_min, weight_max = KL_LOSS_WEIGHT_GRAPH[i]
    # print("{}, {}, {}".format(epochs, weight_min, weight_max))

    def compile_and_train(epochs, initial_epoch):
        vae.save_weights("temp_weights")
        vae.compile(
            optimizer=Adam(LEARNING_RATE),
            loss=['huber_loss', 'mean_squared_error'],
            loss_weights=[100, 5000],
        )
        vae.load_weights("temp_weights")
        vae.fit(
            [x_train_features, x_train_out_pos],
            [y_train_out_pos, y_train_absorbtion],
            epochs=epochs + initial_epoch,
            callbacks=[tensorboard, CustomCallback(vae)],
            initial_epoch=initial_epoch
            )

    if weight_min == weight_max:
        vae.kl_loss_weight = weight_min
        # print("kl_loss_weight is {}".format(vae.kl_loss_weight))
        # print("current_epoch: {}".format(current_epoch))
        compile_and_train(epochs, current_epoch)
        current_epoch += epochs
    else:
        for e in range(0, epochs):
            vae.kl_loss_weight = float(e) / float(epochs) * (weight_max - weight_min) + weight_min
            # print("kl_loss_weight is {}".format(vae.kl_loss_weight))
            # print("current_epoch: {}".format(current_epoch))
            compile_and_train(1, current_epoch)
            current_epoch += 1

vae.save("models/" + NAME)

# if EPOCHS1 > 0:
#     vae.compile(
#         optimizer=Adam(0.002),
#         loss=['huber_loss', 'mean_squared_error'],
#         loss_weights=[100, 5000],
#     )
#     vae.fit(
#         [x_train_features, x_train_out_pos],
#         [y_train_out_pos, y_train_absorbtion],
#         epochs=EPOCHS1,
#         callbacks=[tensorboard, CustomCallback(vae)],
#         )

# vae.save_weights("temp_weights")
# vae.phase1 = False

# if EPOCHS2 > 0:
#     vae.compile(
#         optimizer=Adam(0.002),
#         loss=['huber_loss', 'mean_squared_error'],
#         loss_weights=[100, 5000]
#     )
#     vae.load_weights("temp_weights")
#     vae.fit(
#         [x_train_features, x_train_out_pos],
#         [y_train_out_pos, y_train_absorbtion],
#         epochs=EPOCHS2,
#         callbacks=[tensorboard, CustomCallback(vae)],
#         )

# vae.save("models/" + NAME)
