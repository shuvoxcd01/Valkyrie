from Valkyrie.src.network.q_networks.base_q_network import BaseQNetwork
import tensorflow as tf
from tf_agents.networks import sequential


class CartPoleQNetwork(BaseQNetwork, sequential.Sequential):
    def __init__(self, layers) -> None:
        BaseQNetwork.__init__(self)
        sequential.Sequential.__init__(self, layers=layers)

    def get_mutation_layers(self):
        return [self.layers[-2]]

    def get_crossover_layers(self):
        return [self.layers[-2]]
