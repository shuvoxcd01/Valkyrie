from abc import ABC, abstractmethod
from typing import List
from tf_agents.networks import q_network
import tensorflow as tf


class BaseQNetwork(ABC):
    @abstractmethod
    def get_mutation_layers(self) -> List:
        pass

    @abstractmethod
    def get_crossover_layers(self) -> List:
        pass

    @abstractmethod
    def get_pretrained_layers(self):
        pass

    @abstractmethod
    def set_pretrained_layers(self, layers: List):
        pass
