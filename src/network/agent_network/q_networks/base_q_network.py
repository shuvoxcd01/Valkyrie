from abc import ABC, abstractmethod
from typing import List
from tf_agents.networks import q_network
import tensorflow as tf

from Valkyrie.src.network.pretraining_network.base_pretraining_network import (
    BasePretrainingNetwork,
)


class BaseQNetwork(ABC):
    @abstractmethod
    def get_mutation_layers(self) -> List:
        pass

    @abstractmethod
    def get_crossover_layers(self) -> List:
        pass

    def get_pretraining_network(self):
        return self.pretraining_network

    def set_pretraining_network(self, network: BasePretrainingNetwork):
        self.pretraining_network = network
