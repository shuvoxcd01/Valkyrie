from typing import List
from tf_agents.networks import q_network
import tensorflow as tf

from Valkyrie.src.network.q_networks.base_q_network import BaseQNetwork
import logging


class AtariQNetwork(q_network.QNetwork, BaseQNetwork):
    def __init__(self, input_tensor_spec, action_spec,
                 preprocessing_layers=None,
                 preprocessing_combiner=None,
                 conv_layer_params=None,
                 fc_layer_params=...,
                 dropout_layer_params=None,
                 activation_fn=tf.keras.activations.relu,
                 kernel_initializer=None, batch_squash=True,
                 dtype=tf.float32, name='QNetwork'):

        if kernel_initializer is None:
            logging.info("No explicit initializer provided for AtariQNetwork.")
        else:
            logging.info(
                f"Explicitly provided AtariQNetwork initializer: {kernel_initializer}")

        q_network.QNetwork.__init__(self, input_tensor_spec, action_spec, preprocessing_layers, preprocessing_combiner, conv_layer_params,
                                    fc_layer_params, dropout_layer_params, activation_fn, kernel_initializer, batch_squash, dtype, name)
        BaseQNetwork.__init__(self)

    def call(self, observation, step_type=None, network_state=(), training=False):
        observation = tf.cast(observation, tf.float32)
        observation = observation / 255

        return super(AtariQNetwork, self).call(observation, step_type=step_type, network_state=network_state, training=training)

    def get_mutation_layers(self) -> List:
        embedding_network = self.layers[0]
        return [embedding_network.layers[-1]]

    def get_crossover_layers(self) -> List:
        embedding_network = self.layers[0]
        return [embedding_network.layers[-1]]
