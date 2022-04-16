from abc import ABC, abstractmethod

from agent.meta_agent.meta_agent import MetaAgent


class MetaAgentCopier(ABC):
    @abstractmethod
    def copy_agent(self, meta_agent: MetaAgent, name: str):
        pass
