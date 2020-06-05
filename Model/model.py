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

NAME = "test-{}".format(datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
EPOCHS = 1

intermediate_dim = 64

absorbtion_int_dim = 32
absorbtion_output_dim = 1

encoder_input_shape = (3, )
decoder_input_shape = (68, )
feature_input_shape = (23, )
feature_output_shape = (64, )
latent_dim = 4
decoder_output_dim = 3
latent_shape = (2, )

print("Using tensorflow " + tf.__version__)
tensorboard = TensorBoard(log_dir="logs\\{}".format(NAME))

# load training data
samples_json = open("train_data/samples_2020-5-18--16-25-28.json", "r")
samples = json.load(samples_json)
# print(samples)

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

print("creating model...")

class Sampler(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(layers.Layer):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(name="Encoder", **kwargs)
        self.dense1 = Dense(intermediate_dim, activation="relu", input_shape=encoder_input_shape)
        self.dense2 = Dense(intermediate_dim, activation="relu")
        self.dense3 = Dense(intermediate_dim, activation="relu")
        self.mean = Dense(latent_dim)
        self.log_var = Dense(latent_dim)
        self.sampler = Sampler()

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)

        z_mean = self.mean(x)
        z_log_var = self.log_var(x)
        z = self.sampler((z_mean, z_log_var))
        return z_mean, z_log_var, z

class Decoder(layers.Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(name="Decoder", **kwargs)
        self.dense1 = Dense(intermediate_dim, activation="relu")
        self.dense2 = Dense(intermediate_dim, activation="relu")
        self.dense3 = Dense(intermediate_dim, activation="relu")
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
        self.dense1 = Dense(intermediate_dim, activation="relu")
        self.dense2 = Dense(intermediate_dim, activation="relu")
        self.dense3 = Dense(intermediate_dim, activation="relu")

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

class Absorbtion(layers.Layer):
    def __init__(self, **kwargs):
        super(Absorbtion, self).__init__(name="Absorbtion", **kwargs)
        self.dense1 = Dense(absorbtion_int_dim, activation="relu")
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

    def call(self, inputs):
        feature_inputs, encoder_inputs = inputs
        z_mean, z_log_var, z = self.encoder(encoder_inputs)
        latent_features = self.features(feature_inputs)

        decoder_inputs = layers.concatenate([latent_features, z])
        # decoder_inputs = tf.concat([latent_features, z])
        print(decoder_inputs)

        reconstructed = self.decoder(decoder_inputs)
        absorbtion_result = self.absorbtion(latent_features)

        # Add KL divergence regularization loss.
        # kl_loss = 0.5 * tf.reduce_mean(tf.exp(z_log_var) + tf.square(z_mean) - 4 - z_log_var)

        z_var = tf.exp(z_log_var)
        kl_loss = 0.5 * (tf.reduce_sum(z_var) + tf.reduce_sum(tf.square(z_mean)) - 4 - tf.reduce_prod(z_var))
        # self.add_loss(kl_loss)
        return reconstructed, absorbtion_result

vae = VariationalAutoEncoder()
# optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
# mse_loss_fn = tf.keras.losses.MeanSquaredError()
# loss_metric = tf.keras.metrics.Mean()


#
print("compiling model...")
vae.compile(
    optimizer='adam',
    loss='mean_squared_error',
    # loss='mean_absolute_error',
    # loss='cosine_similarity',
    metrics=['accuracy'],
)
# vae.summary()

#
print("training model...")
vae.fit([x_train1, x_train2], [y_train1, y_train2], epochs=EPOCHS, callbacks=[tensorboard])
vae.save("models/" + NAME)

vae.summary()
plot_model(vae, to_file='models/' + NAME + '/model_plot.png', show_shapes=True, show_layer_names=True, dpi=96*2)
