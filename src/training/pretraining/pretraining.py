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
import os


class Pretraining:
    def __init__(
        self,
        running_pretraining_network: BasePretrainingNetwork,
        stable_pretraining_network: BasePretrainingNetwork,
        replay_buffer_manager: ReplayBufferManager,
        optimizer: tf.keras.optimizers.Optimizer,
        num_iteration: int,
        batch_size: int,
        replay_buffer_table_name: str,
        tf_summary_base_dir: str,
        tau: float = 0.7,
    ) -> None:
        self.running_network = running_pretraining_network
        self.stable_network = stable_pretraining_network
        self.replay_buffer_manager = replay_buffer_manager
        self.optimizer = optimizer
        self.loss_fn = losses.MeanSquaredError()
        self.num_iteration = num_iteration
        self.batch_size = batch_size
        self.replay_buffer_table_name = replay_buffer_table_name
        self.summary_witer_dir = os.path.join(tf_summary_base_dir, "pretraining")
        self.tau = tau

        if not os.path.exists(self.summary_witer_dir):
            os.makedirs(self.summary_witer_dir)

        self.summary_writer_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.summary_witer_dir
        )

        self.last_updated_weights = None

        self.logger = logging.getLogger()
        
        self._initialize()

    def _initialize(self):
        iterator = self.replay_buffer_manager.get_observation_iterator(
            table_name=self.replay_buffer_table_name,
            num_parallel_calls=3,
            batch_size=self.batch_size,
            num_steps=1,
            num_prefetch=3,
        )
        
        dummy_input = next(iterator)
        
        self.stable_network(dummy_input)
        self.running_network(dummy_input)
        
        self.running_network.set_weights(self.stable_network.get_weights())
        
        self.logger.info("Pretraining initialized.")

    def train(self):
        self.logger.info("Starting pretraining ...")
        self.running_network.trainable = True
        self.running_network.compile(optimizer=self.optimizer, loss=self.loss_fn)
        self._train()
        self.running_network.trainable = False
        self.logger.info("Pretraining done...")

    def _train(self):
        if self.last_updated_weights:
            assert self.sanity_check(self.running_network.get_weights())

        iterator = self.replay_buffer_manager.get_observation_iterator(
            table_name=self.replay_buffer_table_name,
            num_parallel_calls=3,
            batch_size=self.batch_size,
            num_steps=1,
            num_prefetch=3,
        )

        for _ in range(self.num_iteration):
            training_data = next(iterator)

            self.running_network.fit(
                training_data,
                training_data,
                epochs=1,
                callbacks=[self.summary_writer_callback],
            )

        self.update_stable_network()
        self.logger.info("Stable network updated.")

        self.last_updated_weights = self.running_network.get_weights()

    def update_stable_network(self):
        for stable_layer, running_layer in zip(
            self.stable_network.layers, self.running_network.layers
        ):
            updated_weights_list = []
            for stable_weights, running_weights in zip(
                stable_layer.get_weights(), running_layer.get_weights()
            ):
                updated_weights = (
                    1.0 - self.tau
                ) * stable_weights + self.tau * running_weights
                updated_weights_list.append(updated_weights)

            stable_layer.set_weights(updated_weights_list)

    def sanity_check(self, current_weights):
        _sum = 0.0
        for prev_layer_w, cur_layer_w in zip(
            self.last_updated_weights, current_weights
        ):
            _sum += tf.reduce_sum(prev_layer_w - cur_layer_w).numpy()

        return _sum == 0.0
