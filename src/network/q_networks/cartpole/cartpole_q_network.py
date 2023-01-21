import types
from typing import Optional, Sequence, Text
from Valkyrie.src.network.q_networks.base_q_network import BaseQNetwork
import tensorflow as tf
from tf_agents.networks import sequential


class CartPoleQNetwork(BaseQNetwork, sequential.Sequential):
    def __init__(self, layers: Sequence[tf.keras.layers.Layer], input_spec=None, name: Optional[Text] = None):
        BaseQNetwork.__init__(self)
        sequential.Sequential.__init__(self, layers, input_spec, name)

    def call(self, inputs, network_state=..., **kwargs):
        return super().call(inputs, network_state, **kwargs)

    def get_mutation_layers(self):
        return [self.layers[-2]]

    def get_crossover_layers(self):
        return [self.layers[-2]]
