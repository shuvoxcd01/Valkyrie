import os
import logging
import shutil
from tf_agents.agents import TFAgent
from tf_agents.utils import common
from checkpoint_manager.checkpoint_manager import CheckpointManager
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
import tensorflow as tf


class ReplayBufferCheckpointManager(CheckpointManager):
    def __init__(self, base_ckpt_dir: str, replay_buffer: ReplayBuffer) -> None:
        super().__init__(base_ckpt_dir=base_ckpt_dir)
        self.replay_buffer = replay_buffer
        self.ckpt_dir = self._get_checkpoint_dir()
        self.checkpointer = self.create_or_initialize_checkpointer()

    def _get_checkpoint_dir(self):
        ckpt_dir = os.path.join(self.base_ckpt_dir, "replay_buffer")

        if os.path.exists(ckpt_dir):
            self.logger.debug("Checkpoint directory already exists.")
        else:
            self.logger.debug("Checkpoint directory created.")

        return ckpt_dir

    def create_or_initialize_checkpointer(self):
        checkpointer = common.Checkpointer(
            ckpt_dir=self.ckpt_dir,
            max_to_keep=1,
            replay_buffer=self.replay_buffer
        )

        self.restore_checkpointer(checkpointer)

        return checkpointer

    def delete_checkpointer(self):
        if not os.path.exists(self.ckpt_dir):
            # ToDo: Raise a specific exception
            raise Exception("Checkpoint does not exist.")

        shutil.rmtree(self.ckpt_dir)

    def save_checkpointer(self):
        global_step = tf.compat.v1.train.get_global_step()
        self.checkpointer.save(global_step=global_step)
