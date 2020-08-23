import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras as K
import numpy as np
import sys
import json

NAME = sys.argv[1]
# NAME = "test-2020-07-15--15-28-22"
# NAME = "test-2020-07-15--21-00-57"
# NAME = "test-2020-07-16--13-58-26"
stddev = float(sys.argv[2])

model_name = ""
weights_name = ""

slash_index = NAME.find("/")
if slash_index != -1:
    model_name = NAME[0:slash_index]
    weights_name = NAME + "/all"
else:
    model_name = NAME
    weights_name = model_name + "/final/all"

print("model: " + model_name)
print("weights: " + weights_name)

vae = K.models.load_model("models/" + model_name)

vae.load_weights("weights/" + weights_name)

input_data = sys.argv[3]
# input_data = "0.0,-0.2617919,-0.6461322,-0.643001139,1.93569016,0.748073936,1.21022677,6.57828951,5.93053341,2.58201337,2.129546,21.522665,-5.03728151,-19.9051876,-13.8902283,0.221121177,-5.47294331,-31.72459,-16.8324146,-1.72248054,0.173161089,0.9,1.0"
input_data = np.array([np.array([float(x) for x in input_data.split(',')])])

latent_features = vae.features(input_data)
absorbtion_result = vae.absorbtion(latent_features)

for i in range(0, 100):
    z = tf.keras.backend.random_normal(shape=(1, 4), mean=0.0, stddev=stddev)
    reconstructed = vae.decoder(layers.concatenate([latent_features, z]))

    # result = vae.predict([input_data, input_pos])
    print("# " + str(float(reconstructed[0][0])) + "," + str(float(reconstructed[0][1])) + "," + str(float(reconstructed[0][2])) + "," + str(float(absorbtion_result[0][0])))
