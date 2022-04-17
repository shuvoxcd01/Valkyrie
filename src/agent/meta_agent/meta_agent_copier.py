from abc import ABC, abstractmethod
from fitness_evaluator.fitness_evaluator import FitnessEvaluator

from agent.meta_agent.meta_agent import MetaAgent


class MetaAgentCopier(ABC):
    @abstractmethod
    def copy_agent(self, meta_agent: MetaAgent, name: str):
        pass

    @abstractmethod
    def crossover(self, agent_1: MetaAgent, agent_2: MetaAgent,
                  agent_1_keep_precentage: float, fitness_evaluator: FitnessEvaluator):
                  pass
