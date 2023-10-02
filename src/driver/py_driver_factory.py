from typing import List, Optional
from driver.driver_factory import DriverFactory
from tf_agents.environments import PyEnvironment
from tf_agents.drivers import py_driver
from tf_agents.policies import py_policy
import logging


class PyDriverFactory(DriverFactory):
    def __init__(self, common_observers: Optional[List]) -> None:
        super().__init__(common_observers=common_observers)
        self.logger = logging.getLogger(__file__)

    def _get_driver(
        self,
        env: PyEnvironment,
        observers: List,
        policy: py_policy.PyPolicy,
        max_steps: int,
        **kwargs,
    ):
        max_episodes = kwargs.get("max_episodes", 1)

        self.logger.info(
            f"""Creating dynamic step driver with the following params: 
            env:{env}, 
            observers: {observers}, 
            policy: {policy}, 
            max_steps: {max_steps},
            max_episodes: {max_episodes}"""
        )

        return py_driver.PyDriver(
            env=env,
            policy=policy,
            observers=observers,
            max_steps=max_steps,
            max_episodes=max_episodes,
        )
