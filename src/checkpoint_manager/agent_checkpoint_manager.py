import os
import logging
import shutil
from tf_agents.agents import TFAgent
from tf_agents.utils import common
from checkpoint_manager.checkpoint_manager import CheckpointManager


class AgentCheckpointManager(CheckpointManager):
    def __init__(self, base_ckpt_dir: str) -> None:
        super().__init__(base_ckpt_dir=base_ckpt_dir)

    def create_or_initialize_checkpointer(self, agent: TFAgent):
        ckpt_dir = os.path.join(self.base_ckpt_dir, agent.name)
        if os.path.exists(ckpt_dir):
            self.logger.debug("Checkpoint directory already exists.")
        else:
            self.logger.debug("Checkpoint directory created.")

        checkpointer = common.Checkpointer(
            ckpt_dir=ckpt_dir,
            max_to_keep=1,
            agent=agent,
            policy=agent.policy)

        return checkpointer

    def delete_checkpointer(self, agent: TFAgent):
        ckpt_dir = os.path.join(self.base_ckpt_dir, agent.name)

        if not os.path.exists(ckpt_dir):
            # ToDo: Raise a specific exception
            raise Exception("Checkpoint does not exist.")

        shutil.rmtree(ckpt_dir)
