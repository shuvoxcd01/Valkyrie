from typing import List
from driver.driver_factory import DriverFactory
from tf_agents.environments import PyEnvironment
from tf_agents.drivers import py_driver
from tf_agents.policies import py_policy

class PyDriverFactory(DriverFactory):
    def __init__(self, env:PyEnvironment, observers: List) -> None:
        super().__init__(env, observers)

    def get_driver(self, policy:py_policy.PyPolicy, num_steps:int):
        return py_driver.PyDriver(
            env=self.env,
            policy=policy,
            observers=self.observers,
            max_steps=num_steps
        )
