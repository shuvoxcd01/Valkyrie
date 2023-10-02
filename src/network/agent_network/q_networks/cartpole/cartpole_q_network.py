import types
from typing import List, Optional, Sequence, Text
import tensorflow as tf
from tf_agents.networks import sequential

from Valkyrie.src.network.agent_network.q_networks.base_q_network import BaseQNetwork


class CartPoleQNetwork(BaseQNetwork, sequential.Sequential):
    def __init__(
        self,
        layers: Sequence[tf.keras.layers.Layer],
        input_spec=None,
        name: Optional[Text] = None,
        **kwargs
    ):
        BaseQNetwork.__init__(self)
        sequential.Sequential.__init__(self, layers, input_spec, name)
        trainable_layers_len = kwargs.get("len_trainable_layers", 2)
        self.pretrained_layers = self.layers[:-trainable_layers_len]
        self.trainable_layers = self.layers[-trainable_layers_len:]

    def call(self, inputs, network_state=..., **kwargs):
        return super().call(inputs, network_state, **kwargs)

    def get_mutation_layers(self):
        return self.trainable_layers

    def get_crossover_layers(self):
        return self.trainable_layers

    def get_pretrained_layers(self):
        return self.pretrained_layers

    def set_pretrained_layers(self, layers: List):
        assert len(self.pretrained_layers) == len(layers)

        for i in range(len(self.pretrained_layers)):
            self.pretrained_layers[i].set_weights(layers[i].get_weights())

    def get_pretraining_network(self):
        encoder = tf.keras.Sequential(self.pretrained_layers)
