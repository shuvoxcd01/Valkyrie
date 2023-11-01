from typing import List, Optional
from singleton_decorator import singleton
import tensorflow as tf
import logging
import math
from Valkyrie.src.network.pretraining_network.base_pretraining_network import (
    BasePretrainingNetwork,
)


class CartPolePretrainingNetwork(BasePretrainingNetwork):
    def __init__(
        self,
        input_tensor_spec,
        encoder_fc_layer_params,
        decoder_fc_layer_params,
        *args,
    ):
        super().__init__()

        self.encoder_fc_layer_params = encoder_fc_layer_params
        self.decoder_fc_layer_params = decoder_fc_layer_params

        self.decoder_output_shape = input_tensor_spec.shape

        self.encoder_network, self.encoder_layers = self._build_encoder()
        self.decoder_network = self._build_decoder()

    def get_encoder_layers(self):
        return self.encoder_layers

    def create_dense_layer(self, num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode="fan_in", distribution="truncated_normal"
            ),
        )

    def _build_encoder(self, kernel_initializer=None):
        if kernel_initializer is None:
            logging.info("No explicit initializer provided for AtariQNetwork.")
        else:
            logging.info(f"Explicitly provided initializer: {kernel_initializer}")

        if kernel_initializer is None:
            kernel_initializer = tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03
            )

        dense_layers = [
            self.create_dense_layer(num_units)
            for num_units in self.encoder_fc_layer_params
        ]

        return tf.keras.Sequential(dense_layers), dense_layers

    def _build_decoder(self):
        decoder_head = [
            tf.keras.layers.Dense(tf.math.reduce_prod(self.decoder_output_shape)),
            tf.keras.layers.Reshape(self.decoder_output_shape),
        ]

        dense_layers = [
            self.create_dense_layer(num_units)
            for num_units in self.decoder_fc_layer_params
        ]

        decoder = tf.keras.Sequential(dense_layers + decoder_head)

        return decoder

    def call(self, x):
        encoded = self.encoder_network(x)
        decoded = self.decoder_network(encoded)
        return decoded
