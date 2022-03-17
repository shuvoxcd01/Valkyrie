from abc import ABC, abstractmethod
from typing import List


class DriverFactory(ABC):
    @abstractmethod
    def get_driver(self, policy, num_steps):
        pass
