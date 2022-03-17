from abc import ABC, abstractmethod


class AgentFactory(ABC):
    @abstractmethod
    def get_agent(self):
        pass