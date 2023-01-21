from Valkyrie.src.agent.meta_agent.meta_agent_factory import MetaAgentFactory
from tf_agents.agents import TFAgent
from summary_writer.summary_writer_manager import SummaryWriterManager
from checkpoint_manager.agent_checkpoint_manager import AgentCheckpointManager
from agent.meta_agent.meta_q_agent.meta_q_agent import MetaQAgent


class MetaQAgentFactory(MetaAgentFactory):

    def get_agent(self, tf_agent: TFAgent, checkpoint_manager: AgentCheckpointManager, summary_writer_manager: SummaryWriterManager, agent_copier: "MetaQAgentCopier", fitness=0, previous_fitness=0, generation: int = 0, name=None):
        agent = MetaQAgent(tf_agent=tf_agent, checkpoint_manager=checkpoint_manager,
                           summary_writer_manager=summary_writer_manager, agent_copier=agent_copier, fitness=fitness,
                           previous_fitness=previous_fitness, generation=generation, name=name)

        return agent
