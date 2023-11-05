from tf_agents.drivers.driver import Driver
from typing import List, Optional, Union
import tensorflow as tf
from Valkyrie.src.agent.meta_agent.meta_agent import MetaAgent
from Valkyrie.src.checkpoint_manager.pretraining_encoder_decoder_network_checkpoint_manager import (
    PretrainingNetworkCheckpointManager,
)
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

from Valkyrie.src.replay_buffer.unified_replay_buffer.unified_replay_buffer_manager import (
    UnifiedReplayBufferManager,
)


class Pretraining:
    def __init__(
        self,
        running_pretraining_network: BasePretrainingNetwork,
        stable_pretraining_network: BasePretrainingNetwork,
        replay_buffer_manager: UnifiedReplayBufferManager,
        optimizer: tf.keras.optimizers.Optimizer,
        num_iteration: int,
        batch_size: int,
        tf_summary_base_dir: str,
        stable_network_checkpoint_manager: PretrainingNetworkCheckpointManager,
        tau: float = 0.5,
        stable_network_update_period=500,
    ) -> None:
        self.running_network = running_pretraining_network
        self.stable_network = stable_pretraining_network
        self.replay_buffer_manager = replay_buffer_manager
        self.optimizer = optimizer
        self.loss_fn = losses.MeanSquaredError()
        self.num_iteration = num_iteration
        self.batch_size = batch_size
        self.summary_witer_dir = os.path.join(tf_summary_base_dir, "pretraining")
        self.tau = tau
        self.stable_network_update_period = stable_network_update_period
        self.train_iter_counter = 0

        self.stable_network_checkpoint_manager = stable_network_checkpoint_manager
        self.stable_network_checkpoint_manager.create_or_initialize_checkpointer()

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

    def train(self, agents_to_sync: Optional[List[MetaAgent]] = None):
        self.logger.info("Starting pretraining ...")
        self.running_network.trainable = True
        self.running_network.compile(optimizer=self.optimizer, loss=self.loss_fn)

        self._train(agents_to_sync=agents_to_sync)

        self.running_network.trainable = False
        self.logger.info("Pretraining done...")

    def sync_agent(self, agent: MetaAgent):
        network = agent.get_network()
        network.set_pretraining_network(self.stable_network.get_pretraining_network())

    def train_iter_counter_updater_callback(self):
        self.train_iter_counter += 1

    def _train(self, agents_to_sync: Optional[List[MetaAgent]] = None):
        if self.last_updated_weights:
            assert self.sanity_check(self.running_network.get_weights())

        iterator = self.replay_buffer_manager.get_observation_iterator(
            num_parallel_calls=3,
            batch_size=self.batch_size,
            num_steps=1,
            num_prefetch=3,
        )

        for _ in range(self.num_iteration):
            training_data = next(iterator)
            training_data = tf.cast(training_data, tf.float32)

            X = training_data
            Y = training_data / 255.0

            self.running_network.fit(
                X,
                Y,
                epochs=1,
                callbacks=[self.summary_writer_callback],
            )
            self.train_iter_counter += 1

            if self.train_iter_counter % self.stable_network_update_period == 0:
                self.update_stable_network()
                self.logger.info("Stable network updated.")

                if agents_to_sync:
                    for agent in agents_to_sync:
                        self.logger.info(
                            f"Synchronizing agent {agent.name}'s pretrained layers."
                        )
                        self.sync_agent(agent)
                        self.logger.info(f"Agent {agent.name} synced.")

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

        self.stable_network_checkpoint_manager.save_checkpointer()

    def sanity_check(self, current_weights):
        _sum = 0.0
        for prev_layer_w, cur_layer_w in zip(
            self.last_updated_weights, current_weights
        ):
            _sum += tf.reduce_sum(prev_layer_w - cur_layer_w).numpy()

        return _sum == 0.0
