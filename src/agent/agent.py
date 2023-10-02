from abc import ABC, abstractmethod
from tf_agents.drivers.driver import Driver
from tf_agents.agents import TFAgent
from summary_writer.summary_writer_manager import SummaryWriterManager
from checkpoint_manager.agent_checkpoint_manager import AgentCheckpointManager


class Agent(ABC):
    def __init__(
        self, name: str, network, fitness=0, previous_fitness=0, generation: int = 0
    ) -> None:
        self.name = name
        self.previous_fitness = previous_fitness
        self.fitness = fitness
        self.generation = generation
        self.network = network

    def update_fitness(self, value):
        self.previous_fitness = self.fitness
        self.fitness = value

    @abstractmethod
    def mutate(self, mean: float, variance: float):
        pass

    @abstractmethod
    def crossover(self, partner, self_keep_percentage):
        pass

    @abstractmethod
    def copy(self):
        pass

    @abstractmethod
    def get_network(self):
        pass
