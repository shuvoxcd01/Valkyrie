from agent.meta_agent.meta_agent_copier import MetaAgentCopier
from agent.meta_agent.meta_q_agent.meta_q_agent import MetaQAgent
from driver.driver_factory import DriverFactory
from agent.tf_agent.agent_factory import AgentFactory
from checkpoint_manager.agent_checkpoint_manager_factory import AgentCheckpointManagerFactory
from summary_writer.summay_writer_manager_factory import SummaryWriterManagerFactory
import tensorflow as tf
from tf_agents.networks import Network, Sequential
from tf_agents.policies import random_tf_policy, random_py_policy, py_tf_eager_policy


class MetaQAgentCopier(MetaAgentCopier):
    def __init__(self, agent_factory: AgentFactory,
                 agent_ckpt_manager_factory: AgentCheckpointManagerFactory,
                 summary_writer_manager_factory: SummaryWriterManagerFactory,
                 max_collect_steps: int,
                 max_collect_episodes: int = 1) -> None:
        self.agent_factory = agent_factory
        self.agent_ckpt_manager_factory = agent_ckpt_manager_factory
        self.summary_writer_manager_factory = summary_writer_manager_factory
        self.max_collect_steps = max_collect_steps
        self.max_collect_episodes = max_collect_episodes

    def copy_agent(self, meta_agent: MetaQAgent, name: str):
        q_net = meta_agent.tf_agent._q_network.copy()

        if isinstance(q_net, Network) and not isinstance(q_net, Sequential):
            q_net.create_variables()
            q_net.set_weights(meta_agent.tf_agent._q_network.get_weights())

        learning_rate = meta_agent.tf_agent._optimizer.learning_rate.numpy()
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate)

        training_step_counter = tf.Variable(
            meta_agent.tf_agent.train_step_counter.numpy())

        copied_tf_agent = self.agent_factory.get_agent(
            name=name, network=q_net, optimizer=optimizer, train_step_counter=training_step_counter)

        agent_checkpoint_manager = self.agent_ckpt_manager_factory.get_agent_checkpoint_manager(
            agent=copied_tf_agent)
        summary_writer_manager = self.summary_writer_manager_factory.get_summary_writer_manager(
            tf_agent=copied_tf_agent)

        fitness = meta_agent.fitness
        previous_fitness = meta_agent.previous_fitness
        tweak_probability = meta_agent.tweak_probability

        copied_meta_agent = MetaQAgent(
            tf_agent=copied_tf_agent, checkpoint_manager=agent_checkpoint_manager,
            summary_writer_manager=summary_writer_manager,
            fitness=fitness, previous_fitness=previous_fitness,
            tweak_probability=tweak_probability)

        copied_meta_agent.checkpoint_manager.save_checkpointer()

        return copied_meta_agent
