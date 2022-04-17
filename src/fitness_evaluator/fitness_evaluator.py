from typing import Union
from tf_agents.environments import TFEnvironment, PyEnvironment
from tf_agents.policies import py_policy, tf_policy, py_tf_eager_policy


class FitnessEvaluator:
    def __init__(self, environment: Union[TFEnvironment, PyEnvironment], num_episodes: int = 10) -> None:
        self.environment = environment
        self.num_episodes = num_episodes

    def evaluate_fitness(self, policy: Union[py_policy.PyPolicy, tf_policy.TFPolicy]):
        if isinstance(self.environment, TFEnvironment):
            assert isinstance(policy, tf_policy.TFPolicy)
        elif isinstance(self.environment, PyEnvironment):
            assert isinstance(policy, py_policy.PyPolicy)
        else:
            raise Exception  # ToDo (raise appropriate exception)

        total_return = 0.0

        for _ in range(self.num_episodes):
            time_step = self.environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = self.environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / self.num_episodes

        if isinstance(policy, tf_policy.TFPolicy):
            avg_return = avg_return.numpy()[0]

        return avg_return
