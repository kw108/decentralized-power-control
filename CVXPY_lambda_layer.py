import tensorflow.keras.layers
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.datasets
import tensorflow.keras.utils
import tensorflow.keras.backend
import numpy as np
import tensorflow as tf

from synthesize_channel_gains import RandGenerator


# set global parameters
num_ues = 4
num_aps = 4
noise_dbm = -60
filename = 'channel_gain.mat'
fading_type = 'rayleigh_gain'
num_training_samples = 900  # number of channel gain and user rate samples in training set
num_test_samples = 100  # number of channel gain and user rate samples in test set


# generate training and test data
rand_generator = RandGenerator(num_ues, num_aps, filename)
gains_train, ue_rates_train = rand_generator.generate(num_training_samples, fading_type, offset=0)
gains_test, ue_rates_test = rand_generator.generate(num_test_samples, fading_type, offset=num_training_samples)

# generate random training and test data
g_train = tf.constant(gains_train)
r_train = tf.constant(ue_rates_train)

g_test = tf.constant(gains_test)
r_test = tf.constant(ue_rates_test)

# generate constant noise
v_train = tf.constant(np.ones(shape=(num_training_samples, num_aps, num_ues)) * 10 ** (noise_dbm / 10))
v_test = tf.constant(np.ones(shape=(num_test_samples, num_aps, num_ues)) * 10 ** (noise_dbm / 10))

# generate all-zero targets since we want to minimize total power
zero_train = tf.constant(np.zeros(shape=(num_training_samples, num_aps * num_ues)))
zero_test = tf.constant(np.zeros(shape=(num_test_samples, num_aps * num_ues)))


# input_layer_g, input_layer_r, input_layer_v
input_layer_g = tensorflow.keras.layers.Input(shape=g_train[0].shape, name="input_layer_g")
input_layer_r = tensorflow.keras.layers.Input(shape=r_train[0].shape, name="input_layer_r")
input_layer_v = tensorflow.keras.layers.Input(shape=v_train[0].shape, name="input_layer_v")

# data rate allocation
flatten_layer = tensorflow.keras.layers.Flatten(name="flatten_layer")(input_layer_g)
concatenate_layer = tensorflow.keras.layers.Concatenate(name="concatenate_layer")([flatten_layer, input_layer_r])
weight_layer = tensorflow.keras.layers.Dense(num_aps * num_ues, name="dense_layer")(concatenate_layer)
reshape_layer = tensorflow.keras.layers.Reshape((num_ues, num_aps), name="reshape_layer")(weight_layer)
softmax_layer = tensorflow.keras.layers.Softmax(axis=1, name="softmax_layer")(reshape_layer)
rate_layer = tf.linalg.matmul(tf.linalg.diag(input_layer_r), softmax_layer)
rate_layer_1 = tensorflow.keras.activations.exponential(softmax_layer)
rate_layer_2 = tensorflow.keras.layers.Subtract(name="rate_layer_2")([rate_layer_1, tf.ones_like(rate_layer_1)])

# fixed point iteration for access point 1
d_layer_1 = tf.linalg.matmul(tf.linalg.diag(rate_layer_2[:, :, 0]),
                             tf.linalg.inv(tf.linalg.diag(tf.linalg.diag_part(input_layer_g[:, 0]))))
dot_layer_11 = tf.linalg.matmul(input_layer_g[:, 0] - tf.linalg.diag(tf.linalg.diag_part(input_layer_g[:, 0])),
                                input_layer_v[:, 0][..., tf.newaxis])
add_layer_1 = tensorflow.keras.layers.Add(name="add_layer_1")([dot_layer_11, input_layer_v[:, 0][..., tf.newaxis]])
dot_layer_12 = tf.reduce_sum(tf.linalg.matmul(d_layer_1, add_layer_1), axis=-1)

# fixed point iteration for access point 2
d_layer_2 = tf.linalg.matmul(tf.linalg.diag(rate_layer_2[:, :, 1]),
                             tf.linalg.inv(tf.linalg.diag(tf.linalg.diag_part(input_layer_g[:, 1]))))
dot_layer_21 = tf.linalg.matmul(input_layer_g[:, 1] - tf.linalg.diag(tf.linalg.diag_part(input_layer_g[:, 1])),
                                input_layer_v[:, 1][..., tf.newaxis])
add_layer_2 = tensorflow.keras.layers.Add(name="add_layer_2")([dot_layer_21, input_layer_v[:, 1][..., tf.newaxis]])
dot_layer_22 = tf.reduce_sum(tf.linalg.matmul(d_layer_2, add_layer_2), axis=-1)

# fixed point iteration for access point 3
d_layer_3 = tf.linalg.matmul(tf.linalg.diag(rate_layer_2[:, :, 2]),
                             tf.linalg.inv(tf.linalg.diag(tf.linalg.diag_part(input_layer_g[:, 2]))))
dot_layer_31 = tf.linalg.matmul(input_layer_g[:, 2] - tf.linalg.diag(tf.linalg.diag_part(input_layer_g[:, 2])),
                                input_layer_v[:, 2][..., tf.newaxis])
add_layer_3 = tensorflow.keras.layers.Add(name="add_layer_3")([dot_layer_31, input_layer_v[:, 2][..., tf.newaxis]])
dot_layer_32 = tf.reduce_sum(tf.linalg.matmul(d_layer_3, add_layer_3), axis=-1)

# fixed point iteration for access point 4
d_layer_4 = tf.linalg.matmul(tf.linalg.diag(rate_layer_2[:, :, 3]),
                             tf.linalg.inv(tf.linalg.diag(tf.linalg.diag_part(input_layer_g[:, 3]))))
dot_layer_41 = tf.linalg.matmul(input_layer_g[:, 3] - tf.linalg.diag(tf.linalg.diag_part(input_layer_g[:, 3])),
                                input_layer_v[:, 3][..., tf.newaxis])
add_layer_4 = tensorflow.keras.layers.Add(name="add_layer_4")([dot_layer_41, input_layer_v[:, 3][..., tf.newaxis]])
dot_layer_42 = tf.reduce_sum(tf.linalg.matmul(d_layer_4, add_layer_4), axis=-1)

# power allocation
power_layer = tensorflow.keras.layers.Concatenate(axis=1, name="power_layer")(
    [dot_layer_12, dot_layer_22, dot_layer_32, dot_layer_42])

# end-to-end model
model = tensorflow.keras.models.Model([input_layer_g, input_layer_r, input_layer_v], power_layer, name="model")

model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001), loss="mean_absolute_error")

model.summary()

model.fit([g_train, r_train, v_train], zero_train, epochs=1000, batch_size=64,
          validation_data=([g_test, r_test, v_test], zero_test))
