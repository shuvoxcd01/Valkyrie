from abc import ABC, abstractmethod
from typing import List, Optional, Union
from tf_agents.environments import PyEnvironment, TFEnvironment


class DriverFactory(ABC):
    def __init__(self, common_observers: Optional[List] = None) -> None:
        self.common_observers = common_observers

    def get_driver(
        self,
        env: Union[PyEnvironment, TFEnvironment],
        observers: List,
        policy,
        max_steps: int,
    ):
        if self.common_observers:
            observers += self.common_observers
        observers = list(set(observers))

        return self._get_driver(
            env=env, observers=observers, policy=policy, max_steps=max_steps
        )

    @abstractmethod
    def _get_driver(
        self,
        env: Union[PyEnvironment, TFEnvironment],
        observers: List,
        policy,
        max_steps: int,
    ):
        pass
