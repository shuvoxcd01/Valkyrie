from abc import ABC, abstractmethod
from tf_agents.environments import suite_atari, suite_gym, tf_py_environment, batched_py_environment, parallel_py_environment


class EnvFactory(ABC):

    @abstractmethod
    def get_py_env(self):
        pass

    def get_tf_env(self):
        tf_env = tf_py_environment.TFPyEnvironment(self.get_py_env())

        return tf_env