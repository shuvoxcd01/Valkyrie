from abc import ABC, abstractmethod
from typing import List, Union
from tf_agents.environments import PyEnvironment, TFEnvironment


class DriverFactory(ABC):
    def __init__(self, env: Union[PyEnvironment, TFEnvironment], observers: List) -> None:
        self.env = env
        self.observers = observers

    @abstractmethod
    def get_driver(self, policy, num_steps):
        pass
