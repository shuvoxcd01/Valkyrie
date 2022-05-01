import logging
from agent.meta_agent.meta_agent import MetaAgent
from fitness_evaluator.fitness_evaluator import FitnessEvaluator
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
        self.logger = logging.getLogger()

    def copy_agent(self, meta_agent: MetaQAgent, name: str):
        q_net = meta_agent.tf_agent._q_network.copy()

        if isinstance(q_net, Network) and not isinstance(q_net, Sequential):
            q_net.create_variables()
            q_net.set_weights(meta_agent.tf_agent._q_network.get_weights())

        optimizer = tf.keras.optimizers.Adam.from_config(
            meta_agent.tf_agent._optimizer.get_config())

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
        generation = meta_agent.generation

        copied_meta_agent = MetaQAgent(
            tf_agent=copied_tf_agent, checkpoint_manager=agent_checkpoint_manager,
            summary_writer_manager=summary_writer_manager,
            fitness=fitness, previous_fitness=previous_fitness,
            generation=generation)

        copied_meta_agent.checkpoint_manager.save_checkpointer()

        return copied_meta_agent

    def crossover(self, agent_1: MetaAgent, agent_2: MetaAgent,
                  agent_1_keep_precentage: float, fitness_evaluator: FitnessEvaluator):
        assert type(agent_1) == type(
            agent_2), "Types of crossover partners don't match."

        generation = agent_1.generation + 1
        name = agent_1.tf_agent.name.split(
            "_generation_")[0] + "_generation_" + str(generation)

        q_net = agent_1.tf_agent._q_network.copy()

        if isinstance(q_net, Network) and not isinstance(q_net, Sequential):
            q_net.create_variables()
            q_net.set_weights(
                agent_1.tf_agent._q_network.get_weights())

        agent_2_q_net = agent_2.tf_agent._q_network

        # ToDo: Find a more efficient way to do crossover of weights
        for i in range(len(q_net.layers)):
            if q_net.layers[i].trainable:
                partner_1_weights_list = q_net.layers[i].get_weights()
                partner_2_weights_list = agent_2_q_net.layers[i].get_weights()

                assert len(partner_1_weights_list) == len(
                    partner_2_weights_list)

                new_weights_list = []

                for j in range(len(partner_1_weights_list)):
                    new_weights = agent_1_keep_precentage * \
                        partner_1_weights_list[j] + \
                        (1-agent_1_keep_precentage)*partner_2_weights_list[j]

                    new_weights_list.append(new_weights)

                q_net.layers[i].set_weights(new_weights_list)

        optimizer = tf.keras.optimizers.Adam.from_config(
            agent_1.tf_agent._optimizer.get_config())

        training_step_counter = tf.Variable(0)

        child_tf_agent = self.agent_factory.get_agent(
            name=name, network=q_net, optimizer=optimizer, train_step_counter=training_step_counter)

        agent_checkpoint_manager = self.agent_ckpt_manager_factory.get_agent_checkpoint_manager(
            agent=child_tf_agent)
        summary_writer_manager = self.summary_writer_manager_factory.get_summary_writer_manager(
            tf_agent=child_tf_agent)

        child_meta_agent = MetaQAgent(
            tf_agent=child_tf_agent, checkpoint_manager=agent_checkpoint_manager,
            summary_writer_manager=summary_writer_manager,
            generation=generation)

        child_meta_agent_fitness = fitness_evaluator.evaluate_fitness(
            policy=child_meta_agent.tf_agent.policy)

        child_meta_agent.update_fitness(child_meta_agent_fitness)

        child_meta_agent.checkpoint_manager.save_checkpointer()

        return child_meta_agent
