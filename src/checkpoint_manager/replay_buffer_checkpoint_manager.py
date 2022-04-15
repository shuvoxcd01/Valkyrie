import os
import logging
import shutil
from tf_agents.agents import TFAgent
from tf_agents.utils import common
from checkpoint_manager.checkpoint_manager import CheckpointManager
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer


class ReplayBufferCheckpointManager(CheckpointManager):
    def __init__(self, base_ckpt_dir:str) -> None:
        super().__init__(base_ckpt_dir=base_ckpt_dir)

    def create_or_initialize_checkpointer(self, replay_buffer:ReplayBuffer):
        ckpt_dir = os.path.join(self.base_ckpt_dir, "replay_buffer")
        if os.path.exists(ckpt_dir):
            self.logger.debug("Checkpoint directory already exists.")
        else:
            self.logger.debug("Checkpoint directory created.")

        checkpointer = common.Checkpointer(
            ckpt_dir=ckpt_dir,
            max_to_keep=1,
            replay_buffer = replay_buffer
            )

        return checkpointer

    def delete_checkpointer(self, arg = None):
        assert arg is None, "delete_checkpointer method in ReplayBufferCheckpointManager expects no argument."
        ckpt_dir = os.path.join(self.base_ckpt_dir, "replay_buffer")

        if not os.path.exists(ckpt_dir):
            # ToDo: Raise a specific exception
            raise Exception("Checkpoint does not exist.")

        shutil.rmtree(ckpt_dir)


