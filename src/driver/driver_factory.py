from abc import ABC, abstractmethod
from typing import List, Optional, Union
from tf_agents.environments import PyEnvironment, TFEnvironment


class DriverFactory(ABC):
    def __init__(
        self, env: Union[PyEnvironment, TFEnvironment], observers: List
    ) -> None:
        self.env = env
        self.observers = observers

    @abstractmethod
    def get_driver(
        self,
        env: Union[PyEnvironment, TFEnvironment],
        observers: List,
        policy,
        max_steps: int,
    ):
        pass
