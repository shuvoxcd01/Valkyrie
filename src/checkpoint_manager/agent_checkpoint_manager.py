import os
import logging
import shutil
from tf_agents.agents import TFAgent
from tf_agents.utils import common
from checkpoint_manager.checkpoint_manager import CheckpointManager


class AgentCheckpointManager(CheckpointManager):
    def __init__(self, base_ckpt_dir: str, agent: TFAgent) -> None:
        super().__init__(base_ckpt_dir=base_ckpt_dir)
        self.agent = agent
        self.ckpt_dir = self._get_checkpoint_dir()
        self.checkpointer = self.create_or_initialize_checkpointer()

    def _get_checkpoint_dir(self):
        ckpt_dir = os.path.join(self.base_ckpt_dir, self.agent.name)
        if os.path.exists(ckpt_dir):
            self.logger.debug("Checkpoint directory already exists.")
        else:
            self.logger.debug("Checkpoint directory created.")

        return ckpt_dir

    def create_or_initialize_checkpointer(self):
        checkpointer = common.Checkpointer(
            ckpt_dir=self.ckpt_dir,
            max_to_keep=1,
            agent=self.agent,
            policy=self.agent.policy)

        self.restore_checkpointer(checkpointer)

        return checkpointer

    def delete_checkpointer(self):
        if not os.path.exists(self.ckpt_dir):
            # ToDo: Raise a specific exception
            raise Exception("Checkpoint does not exist.")

        shutil.rmtree(self.ckpt_dir)

    def save_checkpointer(self):
        agent_step = self.agent.train_step_counter.numpy()
        self.checkpointer.save(global_step=agent_step)
