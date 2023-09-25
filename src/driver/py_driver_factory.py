from typing import List, Optional
from driver.driver_factory import DriverFactory
from tf_agents.environments import PyEnvironment
from tf_agents.drivers import py_driver
from tf_agents.policies import py_policy


class PyDriverFactory(DriverFactory):
    def __init__(self, common_observers: Optional[List]) -> None:
        super().__init__(common_observers=common_observers)

    def _get_driver(
        self,
        env: PyEnvironment,
        observers: List,
        policy: py_policy.PyPolicy,
        max_steps: int,
        **kwargs
    ):
        max_episodes = kwargs.get("max_episodes", 1)

        return py_driver.PyDriver(
            env=env,
            policy=policy,
            observers=observers,
            max_steps=max_steps,
            max_episodes=max_episodes,
        )
