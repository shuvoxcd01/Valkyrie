from typing import List, Optional
from driver.driver_factory import DriverFactory
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import TFEnvironment
from tf_agents.policies import tf_policy
import logging


class DynamicStepDriverFactory(DriverFactory):
    def __init__(self, env: TFEnvironment, observers: List) -> None:
        super().__init__(env=env, observers=observers)

    def get_driver(self, policy: tf_policy.TFPolicy, max_steps: int):
        return dynamic_step_driver.DynamicStepDriver(
            env=self.env, policy=policy, observers=self.observers, num_steps=max_steps
        )
