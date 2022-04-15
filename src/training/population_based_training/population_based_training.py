from curses import meta
import imp
import tensorflow as tf
from pickletools import optimize
from typing import List, Union
from tf_agents.agents import TFAgent
from driver.driver_factory import DriverFactory
from agent.meta_agent.meta_agent import MetaAgent
from tf_agents.environments import TFEnvironment, PyEnvironment
from agent.tf_agent.agent_factory import AgentFactory

from checkpoint_manager.agent_checkpoint_manager import AgentCheckpointManager
from checkpoint_manager.replay_buffer_checkpoint_manager import ReplayBufferCheckpointManager
from tf_agents.replay_buffers.replay_buffer import ReplayBuffer
from fitness_evaluator.fitness_evaluator import FitnessEvaluator
from tf_agents.policies import random_tf_policy, random_py_policy, py_tf_eager_policy
from training.gradient_based_training.gradient_based_training import GradientBasedTraining


class PopulationBasedTraining:
    def __init__(self, initial_population: List[MetaAgent],
                 agent_ckpt_manager: AgentCheckpointManager,
                 gradient_based_trainer: GradientBasedTraining,
                 agent_factory: AgentFactory,
                 fitness_evaluator: FitnessEvaluator,
                 collect_driver_factory: DriverFactory
                 ) -> None:

        self.population = initial_population
        self.num_individuals = len(self.population)
        self.best = None

        self.agent_ckpt_manager = agent_ckpt_manager

        self.gradient_based_trainer = gradient_based_trainer
        self.agent_factory = agent_factory
        self.fitness_evaluator = fitness_evaluator
        self.collect_driver_factory = collect_driver_factory

    def assess_fitness(self, meta_agent: MetaAgent) -> float:
        policy = meta_agent.tf_agent.policy
        fitness_value = self.fitness_evaluator.evaluate_fitness(policy=policy)

        return fitness_value

    def train(self):
        for meta_agent in self.population:
            agent_checkpointer = self.agent_ckpt_manager.create_or_initialize_checkpointer(
                meta_agent.tf_agent)
            meta_agent.previous_fitness = meta_agent.fitness

            meta_agent = self.train_with_gradient_based_approach(
                meta_agent, agent_checkpointer)

            meta_agent.fitness = self.assess_fitness(meta_agent=meta_agent)

            if self.best is None or (meta_agent.fitness >= self.best.fitness):
                # ToDo: Do proper cloning
                q_net = meta_agent.tf_agent._q_network.copy()
                q_net.create_variables()
                q_net.set_weights(meta_agent.tf_agent._q_network.get_weights())
                learning_rate = meta_agent.tf_agent._optimizer.learning_rate.numpy()
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=learning_rate)
                training_step_counter = tf.Variable(
                    meta_agent.tf_agent.train_step_counter.numpy())
                best_tf_agent = self.agent_factory.get_agent(
                    name="best", network=q_net, optimizer=optimizer, train_step_counter=training_step_counter)
                best_meta_agent = MetaAgent(tf_agent=best_tf_agent)
                best_meta_agent.fitness = meta_agent.fitness
                best_meta_agent.previous_fitness = meta_agent.previous_fitness
                best_meta_agent.tweak_probability = meta_agent.tweak_probability
                self.best = best_meta_agent

        return self.best

    def train_with_gradient_based_approach(self, meta_agent: MetaAgent, agent_checkpointer):
        collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            meta_agent.tf_agent.collect_policy, use_tf_function=True)
        # ToDo: change num_steps
        collect_driver = self.collect_driver_factory.get_driver(
            policy=collect_policy, num_steps=1)

        meta_agent.tf_agent = self.gradient_based_trainer.train_agent(
            agent=meta_agent.tf_agent, agent_checkpointer=agent_checkpointer, collect_driver=collect_driver)

        return meta_agent
