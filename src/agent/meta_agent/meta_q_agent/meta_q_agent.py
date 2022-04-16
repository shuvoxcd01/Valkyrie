from tf_agents.drivers.driver import Driver
from tf_agents.agents import TFAgent
from agent.meta_agent.meta_agent import MetaAgent
from summary_writer.summary_writer_manager import SummaryWriterManager
from checkpoint_manager.agent_checkpoint_manager import AgentCheckpointManager


class MetaQAgent(MetaAgent):
    def __init__(self, tf_agent: TFAgent, checkpoint_manager: AgentCheckpointManager,
                 summary_writer_manager: SummaryWriterManager,
                 fitness=0, previous_fitness=0,
                 tweak_probability=None) -> None:
        super().__init__(tf_agent=tf_agent, checkpoint_manager=checkpoint_manager, summary_writer_manager=summary_writer_manager,
                         fitness=fitness, previous_fitness=previous_fitness, tweak_probability=tweak_probability)
