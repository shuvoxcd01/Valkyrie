import gym
import numpy as np
from environment.env_factory import EnvFactory
from tf_agents.environments import suite_atari, suite_gym, tf_py_environment, batched_py_environment, parallel_py_environment


class PongFactory(EnvFactory):
    def __init__(self) -> None:
        super().__init__()
        self.env_name = suite_atari.game(
            name='Pong', mode='Deterministic', version='v4')
        self.atari_frame_skip = 4
        self.max_episode_frames = 108000

    def get_py_env(self):
        py_env = suite_atari.load(
            self.env_name,
            max_episode_steps=self.max_episode_frames / self.atari_frame_skip,
            gym_env_wrappers=suite_atari.DEFAULT_ATARI_GYM_WRAPPERS_WITH_STACKING,
            spec_dtype_map={gym.spaces.Box: np.float32}
        )

        return py_env
