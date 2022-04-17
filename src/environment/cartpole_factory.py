from environment.env_factory import EnvFactory
from tf_agents.environments import suite_atari, suite_gym, tf_py_environment, batched_py_environment, parallel_py_environment


class CartPoleFactory(EnvFactory):
    def __init__(self) -> None:
        super().__init__()
        self.env_name = 'CartPole-v0'

    def get_py_env(self):
        py_env = suite_gym.load(environment_name=self.env_name)

        return py_env
