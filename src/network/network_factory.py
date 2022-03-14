from abc import ABC, abstractmethod


class NetworkFactory(ABC):
    @abstractmethod
    def get_network(self):
        pass
