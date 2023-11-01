from typing import List, Optional, Sequence, Text
from tf_agents.networks import q_network
import tensorflow as tf
from tf_agents.networks import sequential

import logging

from Valkyrie.src.network.agent_network.q_networks.base_q_network import BaseQNetwork
from Valkyrie.src.network.pretraining_network.atari.atari_pretraining_network import (
    AtariPretrainingNetwork,
)


class AtariQNetwork(BaseQNetwork, sequential.Sequential):
    def __init__(
        self,
        pretraining_network,
        layers: Sequence[tf.keras.layers.Layer],
        input_spec=None,
        name: Optional[Text] = None,
        **kwargs,
    ):
        BaseQNetwork.__init__(self)
        sequential.Sequential.__init__(self, layers, input_spec, name)

        self.pretraining_network = pretraining_network

    def call(self, inputs, network_state=..., **kwargs):
        inputs = tf.cast(inputs, tf.float32)
        inputs = inputs / 255

        # _shape = observation.shape
        # if _shape == tf.TensorShape([1, 21, 21, 8]):
        #     features = observation
        # else:
        features = self.pretraining_network(inputs)
        output = super().call(features, network_state, **kwargs)

        return output

    def get_mutation_layers(self):
        return self.layers

    def get_crossover_layers(self):
        return self.layers
