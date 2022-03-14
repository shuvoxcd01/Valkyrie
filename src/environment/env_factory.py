from abc import ABC, abstractmethod


class EnvFactory(ABC):

    @abstractmethod
    def get_py_env(self):
        pass

    @abstractmethod
    def get_tf_env(self):
        pass