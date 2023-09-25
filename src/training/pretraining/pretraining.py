from tf_agents.drivers.driver import Driver
from typing import Union
import tensorflow as tf
from Valkyrie.src.checkpoint_manager.replay_buffer_checkpoint_manager import (
    ReplayBufferCheckpointManager,
)
from Valkyrie.src.network.pretraining_network.base_pretraining_network import (
    BasePretrainingNetwork,
)
from tf_agents.environments import TFEnvironment, PyEnvironment
from tensorflow.keras import losses
from Valkyrie.src.replay_buffer.replay_buffer_manager import ReplayBufferManager
import logging


class Pretraining:
    def __init__(
        self,
        pretraining_network: BasePretrainingNetwork,
        # env: Union[TFEnvironment, PyEnvironment],
        replay_buffer_manager: ReplayBufferManager,
        # replay_buffer_checkpoint_manager: ReplayBufferCheckpointManager,
        # initial_collect_driver: Driver,
        optimizer: tf.keras.optimizers.Optimizer,
        num_iteration: int,
        batch_size: int,
        replay_buffer_table_name: str,
    ) -> None:
        self.network = pretraining_network
        # self.env = env
        self.replay_buffer_manager = replay_buffer_manager
        # self.replay_buffer_checkpoint_manager = replay_buffer_checkpoint_manager
        # self.initial_collect_driver = initial_collect_driver
        self.optimizer = optimizer
        self.loss_fn = losses.MeanSquaredError()
        self.num_iteration = num_iteration
        self.batch_size = batch_size
        self.replay_buffer_table_name = replay_buffer_table_name

        self.logger = logging.getLogger()

        # self._initialize()

    # def _initialize(self):
    #     self.logger.debug("Running initial collect driver.")
    #     self.initial_collect_driver.run(self.env.reset())
    #     self.logger.debug("Initial collect driver finished running.")

    def train(self):
        self.network.trainable = True
        self.network.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self._train()
        self.network.trainable = False

    def _train(self):
        observation_dataset = self.replay_buffer_manager.get_observations_as_dataset(
            table_name=self.replay_buffer_table_name,
            num_parallel_calls=3,
            batch_size=self.batch_size,
            num_steps=1,
            num_prefetch=3,
        )

        for _ in range(self.num_iteration):
            dataset_as_list = tf.convert_to_tensor(
                list(observation_dataset.take(1).as_numpy_iterator())[0]
            )

            self.network.fit(
                dataset_as_list,
                dataset_as_list,
                epochs=1,
            )
