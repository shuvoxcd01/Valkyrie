from typing import List, Optional
from singleton_decorator import singleton
import tensorflow as tf
import logging
import math
from Valkyrie.src.network.pretraining_network.base_pretraining_network import (
    BasePretrainingNetwork,
)


class AtariPretrainingNetwork(BasePretrainingNetwork):
    def __init__(
        self,
        input_tensor_spec,
        encoder_fc_layer_params,
        decoder_fc_layer_params,
        encoder_conv_layer_params,
        decoder_conv_layer_params,
        *args,
    ):
        super().__init__()

        self.encoder_fc_layer_params = encoder_fc_layer_params
        self.decoder_fc_layer_params = decoder_fc_layer_params

        self.encoder_conv_layer_params = encoder_conv_layer_params
        self.decoder_conv_layer_params = decoder_conv_layer_params

        self.observation_shape = input_tensor_spec.shape

        self.encoder_network, self.encoder_layers = self._build_encoder()
        self.decoder_network = self._build_decoder()

        self.encoder_last_conv_layer_out_shape = None

    def get_encoder_layers(self):
        return self.encoder_layers

    def _build_encoder(self, kernel_initializer=None):
        encoder_layers = [
            tf.keras.layers.Input(shape=self.observation_shape),
            tf.keras.layers.Conv2D(
                16, (3, 3), activation="relu", padding="same", strides=2
            ),
            tf.keras.layers.Conv2D(
                8, (3, 3), activation="relu", padding="same", strides=2
            ),
        ]
        encoder_network = tf.keras.Sequential(encoder_layers)
        self.encoder_last_conv_layer_out_shape = list(encoder_network.output_shape)[1:]

        encoder_network.add(tf.keras.layers.Flatten())

        return encoder_network, encoder_layers

    def _build_decoder(self):
        decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Reshape(
                    target_shape=self.encoder_last_conv_layer_out_shape
                ),
                tf.keras.layers.Conv2DTranspose(
                    8, kernel_size=3, strides=2, activation="relu", padding="same"
                ),
                tf.keras.layers.Conv2DTranspose(
                    16, kernel_size=3, strides=2, activation="relu", padding="same"
                ),
                tf.keras.layers.Conv2D(
                    self.observation_shape[-1],
                    kernel_size=(3, 3),
                    activation="sigmoid",
                    padding="same",
                ),
            ]
        )

        return decoder

    def call(self, x):
        _shape = x.shape
        assert len(_shape) == 5, "Input observation is expected to have 5 dims."

        batch, step = _shape[0], _shape[1]

        x = tf.reshape(tensor=x, shape=[batch * step, _shape[2], _shape[3], _shape[4]])

        x = tf.cast(x, tf.float32)
        x = x / 25

        encoded = self.encoder_network(x)
        decoded = self.decoder_network(encoded)

        decoded = tf.reshape(tensor=decoded, shape=_shape)

        return decoded
