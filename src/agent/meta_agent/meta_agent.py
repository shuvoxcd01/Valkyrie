from tf_agents.drivers.driver import Driver
from tf_agents.agents import TFAgent
from summary_writer.summary_writer_manager import SummaryWriterManager
from checkpoint_manager.agent_checkpoint_manager import AgentCheckpointManager


class MetaAgent:
    def __init__(self, tf_agent: TFAgent, checkpoint_manager: AgentCheckpointManager,
                 summary_writer_manager: SummaryWriterManager,
                 fitness=0, previous_fitness=0, tweak_probability=None) -> None:
        self.tf_agent = tf_agent
        self.checkpoint_manager = checkpoint_manager
        self.summary_writer_manager = summary_writer_manager
        self.previous_fitness = previous_fitness
        self.fitness = fitness
        self.tweak_probability = tweak_probability

    def update_fitness(self, value):
        self.previous_fitness = self.fitness
        self.fitness = value
