from abc import ABC, abstractmethod
from tf_agents.networks import Network
import tensorflow as tf
from tf_agents.typing.types import Optimizer


class AgentFactory(ABC):
    @abstractmethod
    def get_agent(self, name: str, network: Network, optimizer: Optimizer, train_step_counter: tf.Variable):
        pass
