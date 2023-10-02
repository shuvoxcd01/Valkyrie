from abc import ABC, abstractmethod
from Valkyrie.src.agent.meta_agent.meta_agent import MetaAgent
from Valkyrie.src.fitness_evaluator.fitness_evaluator import FitnessEvaluator


class MetaAgentCopier(ABC):
    @abstractmethod
    def copy_agent(self, meta_agent: MetaAgent, name: str,  agent_generation: int):
        pass
