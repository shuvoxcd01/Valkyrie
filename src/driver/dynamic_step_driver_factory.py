from typing import List
from driver.driver_factory import DriverFactory
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import TFEnvironment
from tf_agents.policies import tf_policy


class DynamicStepDriverFactory(DriverFactory):
    def __init__(self, env: TFEnvironment, observers: List) -> None:
        super().__init__(env=env, observers=observers)

    def get_driver(self, policy: tf_policy.TFPolicy, num_steps: int):
        return dynamic_step_driver.DynamicStepDriver(
            env=self.env,
            policy=policy,
            observers=self.observers,
            num_steps=num_steps
        )
