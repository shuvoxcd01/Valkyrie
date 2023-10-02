import logging
from typing import Optional
from Valkyrie.src.agent.meta_agent.meta_agent_copier import MetaAgentCopier

# from agent.meta_agent.meta_q_agent.meta_q_agent import MetaQAgent
from agent.tf_agent.agent_factory import AgentFactory
from checkpoint_manager.agent_checkpoint_manager_factory import (
    AgentCheckpointManagerFactory,
)
from summary_writer.summay_writer_manager_factory import SummaryWriterManagerFactory
import tensorflow as tf
from tf_agents.networks import Network, Sequential


class MetaQAgentCopier(MetaAgentCopier):
    def __init__(
        self,
        agent_factory: AgentFactory,
        agent_ckpt_manager_factory: AgentCheckpointManagerFactory,
        summary_writer_manager_factory: SummaryWriterManagerFactory,
        meta_agent_factory: "MetaQAgentFactory",
    ) -> None:
        self.agent_factory = agent_factory
        self.agent_ckpt_manager_factory = agent_ckpt_manager_factory
        self.summary_writer_manager_factory = summary_writer_manager_factory
        self.logger = logging.getLogger()
        self.meta_agent_factory = meta_agent_factory

    def copy_agent(
        self,
        meta_agent: "MetaQAgent",
        name: str,
        agent_generation: Optional[int] = None,
        training_step_counter: Optional[int] = None,
        network=None,
        optimizer_name: str = "rmsprop",
        save_checkpoint: bool = True,
    ):
        if network is None:
            network = meta_agent.get_network().copy()

        if isinstance(network, Network) and not isinstance(network, Sequential):
            network.create_variables()

        network.set_weights(meta_agent.tf_agent._q_network.get_weights())

        if optimizer_name != "rmsprop":
            raise Exception("Only RMSProp optimizer supported at the moment.")

        optimizer = tf.compat.v1.train.RMSPropOptimizer(
            learning_rate=2.5e-3,
            decay=0.95,
            momentum=0.0,
            epsilon=0.00001,
            centered=True,
        )

        if training_step_counter is None:
            training_step_counter = tf.Variable(
                meta_agent.tf_agent.train_step_counter.numpy()
            )
        else:
            training_step_counter = tf.Variable(training_step_counter)

        copied_tf_agent = self.agent_factory.get_agent(
            name=name,
            network=network,
            optimizer=optimizer,
            train_step_counter=training_step_counter,
        )

        agent_checkpoint_manager = (
            self.agent_ckpt_manager_factory.get_agent_checkpoint_manager(
                agent=copied_tf_agent
            )
        )
        summary_writer_manager = (
            self.summary_writer_manager_factory.get_summary_writer_manager(
                tf_agent=copied_tf_agent
            )
        )

        fitness = meta_agent.fitness
        previous_fitness = meta_agent.previous_fitness
        generation = (
            agent_generation if agent_generation is not None else meta_agent.generation
        )

        copied_meta_agent = self.meta_agent_factory.get_agent(
            tf_agent=copied_tf_agent,
            checkpoint_manager=agent_checkpoint_manager,
            summary_writer_manager=summary_writer_manager,
            agent_copier=self,
            fitness=fitness,
            previous_fitness=previous_fitness,
            generation=generation,
        )

        if save_checkpoint:
            copied_meta_agent.checkpoint_manager.save_checkpointer()

        return copied_meta_agent
