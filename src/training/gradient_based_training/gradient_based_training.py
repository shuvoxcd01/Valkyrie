from audioop import avg
from tensorboard import summary
from tf_agents.agents import TFAgent
from tf_agents.utils import common
from tf_agents.drivers.driver import Driver
from tf_agents.environments import TFEnvironment, PyEnvironment
from typing import Optional, Union
import tensorflow as tf
from tf_agents.environments import TFEnvironment, PyEnvironment
from tf_agents.policies import py_policy, tf_policy, py_tf_eager_policy
from driver.driver_factory import DriverFactory
from agent.meta_agent.meta_q_agent.meta_q_agent import MetaQAgent
from checkpoint_manager.replay_buffer_checkpoint_manager import ReplayBufferCheckpointManager
from fitness_evaluator.fitness_evaluator import FitnessEvaluator
from replay_buffer.replay_buffer_manager import ReplayBufferManager
import os
from tf_agents.utils.common import Checkpointer
import logging


class GradientBasedTraining:
    def __init__(self, train_env: Union[TFEnvironment, PyEnvironment],
                 replay_buffer_manager: ReplayBufferManager,
                 replay_buffer_checkpoint_manager: ReplayBufferCheckpointManager,
                 initial_collect_driver: Driver,
                 fitness_evaluator: FitnessEvaluator,
                 num_train_iteration: int, log_interval: int, eval_interval: int,
                 collect_driver_factory: DriverFactory,
                 max_collect_steps: int,
                 max_collect_episodes: int = 1,
                 batch_size: int = 32,
                 best_possible_fitness: Optional[int] = None) -> None:

        self.train_env = train_env
        self.replay_buffer_manager = replay_buffer_manager
        self.logger = logging.getLogger()
        self.replay_buffer_checkpoint_manager = replay_buffer_checkpoint_manager
        self.fitness_evaluator = fitness_evaluator
        self.num_train_iteration = num_train_iteration
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.initial_collect_driver = initial_collect_driver
        self.best_possible_fitness = best_possible_fitness
        self.collect_driver_factor = collect_driver_factory
        self.max_collect_steps = max_collect_steps
        self.max_collect_episodes = max_collect_episodes

        self._initialize()

    def _initialize(self):
        self.logger.debug("Running initial collect driver.")
        self.initial_collect_driver.run(self.train_env.reset())
        self.logger.debug("Initial collect driver finished running.")

    def _get_collect_driver(self, tf_agent: TFAgent):
        collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            tf_agent.collect_policy, use_tf_function=True)

        collect_driver = self.collect_driver_factor.get_driver(
            policy=collect_policy, max_steps=self.max_collect_steps, max_episodes=self.max_collect_episodes)

        return collect_driver

    def train_agent(self, meta_agent: MetaQAgent):
        tf_agent = meta_agent.tf_agent
        collect_driver = self._get_collect_driver(tf_agent=tf_agent)

        tf_agent.train = common.function(tf_agent.train)

        iterator = self.replay_buffer_manager.get_dataset_iterator(
            num_parallel_calls=3, batch_size=self.batch_size, num_steps=2, num_prefetch=3)

        for _ in range(self.num_train_iteration):
            collect_driver.run(self.train_env.reset())

            experience, unused_info = next(iterator)
            train_loss = tf_agent.train(experience).loss

            step = tf_agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                self.logger.info(
                    'step = {0}: loss = {1}'.format(step, train_loss))

            if step % self.eval_interval == 0:
                fitness = self.fitness_evaluator.evaluate_fitness(
                    tf_agent.policy)
                meta_agent.update_fitness(fitness)

                self.logger.info('step = {0}: Average Return = {1}'.format(
                    step, fitness))

                meta_agent.summary_writer_manager.write_scalar_summary(
                    name="Average return", data=fitness)

                meta_agent.checkpoint_manager.save_checkpointer()

                if self.best_possible_fitness and meta_agent.fitness >= self.best_possible_fitness:
                    break

        meta_agent.checkpoint_manager.save_checkpointer()

        self.replay_buffer_checkpoint_manager.save_checkpointer()
