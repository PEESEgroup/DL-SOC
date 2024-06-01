import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from keras.models import Model
from functools import partial


def build_model():
    spectral_encoder = keras.Sequential([
        layers.InputLayer(input_shape=(5, 5, 103)),
        layers.Reshape((25, 103, 1)),
        layers.Conv1D(256, 15, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
        layers.Conv1D(256, 15, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
        layers.Conv1D(256, 15, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
        layers.Conv1D(128, 11, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
        layers.Conv1D(128, 11, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
        layers.Conv1D(128, 11, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
        layers.Conv1D(64, 7, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
        layers.Conv1D(64, 7, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
        layers.Conv1D(64, 7, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
        layers.Conv1D(16, 5, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
        layers.Conv1D(1, 3, activation=partial(tf.nn.leaky_relu, alpha=0.1)),
    ])
    
    spatial_encoder = keras.Sequential([
        layers.InputLayer(input_shape=(5, 5, 7)),
        layers.Conv2D(64, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.1), padding='same'),
        layers.Conv2D(128, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.1), padding='same'),
        layers.Conv2D(128, (3, 3), activation=partial(tf.nn.leaky_relu, alpha=0.1), padding='same'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(32, activation='relu')
    ])
    
    regression_net = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(34,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(100, activation='relu'),
        layers.Dense(1, activation='linear')
    ])

    x1 = keras.Input((5, 5, 103))
    o = spectral_encoder(x1)
    o = layers.Reshape((5, 5, 7))(o)
    o = spatial_encoder(o)

    x2 = keras.Input((2,))

    dl_features = layers.Concatenate(axis=-1)([o, x2])

    target = regression_net(dl_features)

    ssl_model = Model(inputs=[x1, x2], outputs=target)
    # print(ssl_model.summary())
    return ssl_model


