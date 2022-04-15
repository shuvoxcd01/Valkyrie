from audioop import avg
from tensorboard import summary
from tf_agents.agents import TFAgent
from tf_agents.utils import common
from tf_agents.drivers.driver import Driver
from tf_agents.environments import TFEnvironment, PyEnvironment
from typing import Union
import tensorflow as tf
from tf_agents.environments import TFEnvironment, PyEnvironment
from tf_agents.policies import py_policy, tf_policy, py_tf_eager_policy
from checkpoint_manager.replay_buffer_checkpoint_manager import ReplayBufferCheckpointManager
from fitness_evaluator.fitness_evaluator import FitnessEvaluator
from replay_buffer.replay_buffer_manager import ReplayBufferManager
import os
from tf_agents.utils.common import Checkpointer
import logging


class GradientBasedTraining:
    def __init__(self, train_env: Union[TFEnvironment, PyEnvironment],
                 replay_buffer_manager: ReplayBufferManager,
                 replay_buffer_checkpointer: Checkpointer,
                 initial_collect_driver: Driver,
                 fitness_evaluator: FitnessEvaluator,
                 num_train_iteration: int, log_interval: int, eval_interval: int,
                 batch_size: int = 32) -> None:

        self.train_env = train_env
        self.replay_buffer_manager = replay_buffer_manager
        self.logger = logging.getLogger()
        self.replay_buffer_checkpointer = replay_buffer_checkpointer
        self.fitness_evaluator = fitness_evaluator
        self.num_train_iteration = num_train_iteration
        self.eval_interval = eval_interval
        self.log_interval = log_interval
        self.batch_size = batch_size
        self.initial_collect_driver = initial_collect_driver

        self._initialize()

    def _initialize(self):
        self.initial_collect_driver.run(self.train_env.reset())

    # def _get_summary_writer(self):
    #     summary_writer_dir = os.path.join(self.base_summary_writer_dir, agent.name)
    #     summary_writer = tf.summary.create_file_writer(summary_writer_dir)

    #     return summary_writer

    def _compute_avg_return(self, policy):
        avg_return = self.fitness_evaluator.evaluate_fitness(policy)

        return avg_return

    def train_agent(self, agent: TFAgent, agent_checkpointer, collect_driver: Driver,
                    summary_writer=None):
        agent.train = common.function(agent.train)

        agent.train_step_counter.assign(0)

        avg_return = self._compute_avg_return(agent.policy)
        returns = [avg_return]

        iterator = self.replay_buffer_manager.get_dataset_iterator(
            num_parallel_calls=3, batch_size=self.batch_size, num_steps=2, num_prefetch=3)

        for _ in range(self.num_train_iteration):
            collect_driver.run(self.train_env.reset())

            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss

            step = agent.train_step_counter.numpy()

            if step % self.log_interval == 0:
                self.logger.info(
                    'step = {0}: loss = {1}'.format(step, train_loss))

            if step % self.eval_interval == 0:
                avg_return = self._compute_avg_return(agent.policy)
                self.logger.info('step = {0}: Average Return = {1}'.format(
                    step, avg_return))
                # with summary_writer.as_default():
                #     tf.summary.scalar("Average return", avg_return, step=step)
                #     summary_writer.flush()
                returns.append(avg_return)

                agent_checkpointer.save(global_step=step)

        # Save the policy at the end of training so that it can be easily deployed.
        # tf_policy_saver.save(policy_dir)
        agent_checkpointer.save(agent.train_step_counter.numpy())

        global_step = tf.compat.v1.train.get_global_step()
        self.replay_buffer_checkpointer.save(global_step=global_step)

        return agent
