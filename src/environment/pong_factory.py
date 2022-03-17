from environment.env_factory import EnvFactory
from tf_agents.environments import suite_atari, suite_gym, tf_py_environment, batched_py_environment, parallel_py_environment


class PongFactory(EnvFactory):
    def __init__(self) -> None:
        super().__init__()
        self.env_name = 'Pong-v0'
        self.atari_frame_skip = 4
        self.max_episode_frames = 108000

    def get_py_env(self):
        py_env = suite_atari.load(
            self.env_name,
            max_episode_steps=self.max_episode_frames / self.atari_frame_skip,
            gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING
        )

        return py_env

    def get_tf_env(self):
        tf_env = tf_py_environment.TFPyEnvironment(self.get_py_env())

        return tf_env
