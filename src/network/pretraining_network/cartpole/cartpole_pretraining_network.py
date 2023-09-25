from singleton_decorator import singleton
import tensorflow as tf
import logging

from Valkyrie.src.network.pretraining_network.base_pretraining_network import (
    BasePretrainingNetwork,
)


@singleton
class CartPolePretrainingNetwork(BasePretrainingNetwork):
    def __init__(self, input_tensor_spec, conv_layer_params, fc_layer_params, *args):
        super().__init__()
        self.decoder_output_shape = input_tensor_spec.shape
        self.fc_layer_params = (128, 256, 256, 128, 50)
        self.decoder_fc_layer_params = None
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
            )
        )

    def _build_encoder(self, kernel_initializer=None):
        if kernel_initializer is None:
            logging.info("No explicit initializer provided for AtariQNetwork.")
        else:
            logging.info(
                f"Explicitly provided AtariQNetwork initializer: {kernel_initializer}"
            )

        if kernel_initializer is None:
            kernel_initializer = tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03
            )

        dense_layers = [
            self.create_dense_layer(num_units) for num_units in self.fc_layer_params
        ]

        return tf.keras.Sequential(dense_layers), dense_layers

    def _build_decoder(self):
        decoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    tf.math.reduce_prod(self.decoder_output_shape), activation="sigmoid"
                ),
                tf.keras.layers.Reshape(self.decoder_output_shape),
            ]
        )

        return decoder

    def call(self, x):
        print(f"Autoencoder called with {x}")
        encoded = self.encoder_network(x)
        decoded = self.decoder_network(encoded)
        return decoded
