import types
from typing import List, Optional, Sequence, Text
import tensorflow as tf
from tf_agents.networks import sequential

from Valkyrie.src.network.agent_network.q_networks.base_q_network import BaseQNetwork
from Valkyrie.src.network.pretraining_network.base_pretraining_network import (
    BasePretrainingNetwork,
)


class CartPoleQNetwork(BaseQNetwork, sequential.Sequential):
    def __init__(
        self,
        pretraining_network: BasePretrainingNetwork,
        layers: Sequence[tf.keras.layers.Layer],
        input_spec=None,
        name: Optional[Text] = None,
        **kwargs
    ):
        BaseQNetwork.__init__(self)
        sequential.Sequential.__init__(self, layers, input_spec, name)
        self.pretraining_network = pretraining_network

    def call(self, inputs, network_state=..., **kwargs):
        features = self.pretraining_network(inputs)
        output = super().call(features, network_state, **kwargs)

        return output

    def get_mutation_layers(self):
        return self.layers

    def get_crossover_layers(self):
        return self.layers
