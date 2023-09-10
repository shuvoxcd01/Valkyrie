import types
from typing import List, Optional, Sequence, Text
from Valkyrie.src.network.q_networks.base_q_network import BaseQNetwork
import tensorflow as tf
from tf_agents.networks import sequential


class CartPoleQNetwork(BaseQNetwork, sequential.Sequential):
    def __init__(self, layers: Sequence[tf.keras.layers.Layer], input_spec=None, name: Optional[Text] = None):
        BaseQNetwork.__init__(self)
        sequential.Sequential.__init__(self, layers, input_spec, name)
        self.pretrained_layers = self.layers[:-2]

    def call(self, inputs, network_state=..., **kwargs):
        return super().call(inputs, network_state, **kwargs)

    def get_mutation_layers(self):
        return [self.layers[-2]]

    def get_crossover_layers(self):
        return [self.layers[-2]]
    
    def get_pretrained_layers(self):
        return self.pretrained_layers
    
    def set_pretrained_layers(self, layers:List):
        assert len(self.pretrained_layers) == len(layers)
        
        for i in range(len(self.pretrained_layers)):
            self.pretrained_layers[i].set_weights(layers[i].get_weights())
