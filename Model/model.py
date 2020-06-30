# import tensorflow.keras as keras
import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import layers
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

NAME = "test-{}".format(datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
EPOCHS = 1000

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
samples_json = open("../train_data/samples_2020-6-24--13-9-50.json", "r")
samples = json.load(samples_json)
# print(samples)

# bring training data into right shape
print("preprocessing input training data...")
x_train_features = np.array([np.array(data['features']) for data in samples])
x_train_out_pos = np.array([np.array(data['out_pos']) for data in samples])
# print(x_train_features)
# print(x_train_out_pos)

print("preprocessing output training data...")
y_train_out_pos = np.array([np.array([data['out_pos']]) for data in samples])
y_train_absorbtion = np.array([np.array([data['absorbtion']]) for data in samples])
# print(y_train_out_pos)
# print(y_train_absorbtion)

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
        latent_features = self.features(feature_inputs)
        z_mean, z_log_var, z = self.encoder(layers.concatenate([latent_features, encoder_inputs]))

        decoder_inputs = layers.concatenate([latent_features, z])
        reconstructed = self.decoder(decoder_inputs)
        absorbtion_result = self.absorbtion(latent_features)

        # Add KL divergence regularization loss.
        # kl_loss = 0.5 * tf.reduce_mean(tf.exp(z_log_var) + tf.square(z_mean) - 4 - z_log_var)

        # print("z_mean shape: ", end="")
        # print(tf.shape(z_mean))
        # print("z_log_var shape: ", end="")
        # print(tf.shape(z_log_var))

        z_var = tf.exp(z_log_var)
        # kl_loss = 0.5 * (tf.reduce_sum(z_var) + tf.tensordot(z_mean, z_mean) - 4 - tf.reduce_sum(z_log_var))
        kl_loss = 0.5 * (tf.reduce_sum(z_var) + tf.reduce_sum(tf.square(z_mean)) - 4 - tf.reduce_sum(z_log_var))
        self.add_loss(kl_loss)
        self.add_metric(kl_loss, aggregation="mean", name="kl_loss")

        return reconstructed, absorbtion_result

vae = VariationalAutoEncoder()

#
print("compiling model...")
vae.compile(
    optimizer=Adam(0.0002),
    loss=['huber_loss', 'mean_squared_error'],
    metrics=['accuracy'],
    loss_weights=[100, 5000]
)
# vae.summary()

#
print("training model...")
vae.fit(
    [x_train_features, x_train_out_pos],
    [y_train_out_pos, y_train_absorbtion],
    epochs=EPOCHS,
    callbacks=[tensorboard],
    )
vae.save("models/" + NAME)

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


vae.summary()
plot_model(vae, to_file='models/' + NAME + '/model_plot.png', show_shapes=True, show_layer_names=True, dpi=96*2)