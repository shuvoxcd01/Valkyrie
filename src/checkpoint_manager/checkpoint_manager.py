from abc import ABC, abstractmethod
import os
import logging
import shutil
from typing import Optional, Union
from tf_agents.agents import TFAgent
from tf_agents.utils import common
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer

class CheckpointManager(ABC):
    def __init__(self, base_ckpt_dir:str) -> None:
        self.base_ckpt_dir = base_ckpt_dir
        self.logger = logging.getLogger()

    @abstractmethod
    def create_or_initialize_checkpointer(self, arg:Union[TFAgent,ReplayBuffer])->common.Checkpointer:
        pass

    @abstractmethod
    def delete_checkpointer(self, arg:Optional[TFAgent]=None)->None:
        pass

    def restore_checkpointer(self, checkpointer: common.Checkpointer):
        return checkpointer.initialize_or_restore()





