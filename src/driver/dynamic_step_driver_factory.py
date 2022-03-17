from typing import List
from driver.driver_factory import DriverFactory
from tf_agents.drivers import dynamic_step_driver


class DynamicStepDriverFactory(DriverFactory):
    def __init__(self, env, observers:List) -> None:
        super().__init__()
        self.env = env
        self.observers = observers

    def get_driver(self, policy, num_steps):
        return dynamic_step_driver.DynamicStepDriver(
            env=self.env,
            policy=policy,
            observers=self.observers,
            num_steps=num_steps
        )
