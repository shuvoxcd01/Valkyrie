from typing import List, Optional
from driver.driver_factory import DriverFactory
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import TFEnvironment
from tf_agents.policies import tf_policy


class DynamicStepDriverFactory(DriverFactory):
    def __init__(self, common_observers: Optional[List] = None) -> None:
        super().__init__(common_observers=common_observers)

    def _get_driver(
        self,
        env: TFEnvironment,
        observers: List,
        policy: tf_policy.TFPolicy,
        max_steps: int,
    ):
        return dynamic_step_driver.DynamicStepDriver(
            env=env, policy=policy, observers=observers, num_steps=max_steps
        )
