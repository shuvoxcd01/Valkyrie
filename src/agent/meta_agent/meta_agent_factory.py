from abc import ABC, abstractmethod
from tf_agents.agents import TFAgent
from summary_writer.summary_writer_manager import SummaryWriterManager
from checkpoint_manager.agent_checkpoint_manager import AgentCheckpointManager

class MetaAgentFactory(ABC):
    @abstractmethod
    def get_agent(self, tf_agent: TFAgent, checkpoint_manager: AgentCheckpointManager,
                 summary_writer_manager: SummaryWriterManager,
                 fitness=0, previous_fitness=0,
                 generation: int = 0, name=None):
        pass
    