from abc import ABC, abstractmethod


class EvolutionaryOperations(ABC):
    @staticmethod
    @abstractmethod
    def crossover():
        pass

    @staticmethod
    @abstractmethod
    def mutate():
        pass
