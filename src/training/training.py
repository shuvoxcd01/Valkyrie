from tf_agents.agents import TFAgent
from tf_agents.utils import common
from tf_agents.drivers.driver import Driver
from tf_agents.environments import TFEnvironment, PyEnvironment
from typing import Union
import tensorflow as tf
from tf_agents.environments import TFEnvironment, PyEnvironment
from tf_agents.policies import py_policy, tf_policy, py_tf_eager_policy
from replay_buffer.replay_buffer_manager import ReplayBufferManager


class Training:
    def __init__(self, agent: TFAgent, collect_driver: Driver, train_env: Union[TFEnvironment, PyEnvironment],
                 eval_env: Union[TFEnvironment, PyEnvironment], replay_buffer_manager: ReplayBufferManager,
                 logdir: str, train_checkpointer) -> None:
        self.agent = agent
        self.collect_driver = collect_driver
        self.train_env = train_env
        self.eval_env = eval_env
        self.replay_buffer_manager = replay_buffer_manager
        self.logger = tf.get_logger()
        self.summary_writer = tf.summary.create_file_writer(logdir)
        self.train_checkpointer = train_checkpointer

    def compute_avg_return(self, environment: PyEnvironment, policy: Union[py_policy.PyPolicy, tf_policy.TFPolicy], num_episodes: int = 10):
        total_return = 0.0

        for _ in range(num_episodes):
            time_step = environment.reset()
            episode_return = 0.0

            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
            total_return += episode_return

        avg_return = total_return / num_episodes

        if isinstance(policy, tf_policy.TFPolicy):
            avg_return = avg_return.numpy()[0]

        return avg_return

    def train_agent(self, num_iterations: int, num_eval_episodes: int, log_interval: int, eval_interval: int, batch_size: int = 32):
        self.agent.train = common.function(self.agent.train)

        self.agent.train_step_counter.assign(0)

        avg_return = self.compute_avg_return(
            self.eval_env, self.agent.policy, num_eval_episodes)
        returns = [avg_return]

        iterator = self.replay_buffer_manager.get_dataset_iterator(
            num_parallel_calls=3, batch_size=batch_size, num_steps=2, num_prefetch=3)

        for _ in range(num_iterations):
            self.collect_driver.run(self.train_env.reset())

            experience, unused_info = next(iterator)
            train_loss = self.agent.train(experience).loss

            step = self.agent.train_step_counter.numpy()

            if step % log_interval == 0:
                self.logger.info(
                    'step = {0}: loss = {1}'.format(step, train_loss))

            if step % eval_interval == 0:
                avg_return = self.compute_avg_return(
                    self.eval_env, self.agent.policy, num_eval_episodes)
                self.logger.info('step = {0}: Average Return = {1}'.format(
                    step, avg_return))
                with self.summary_writer.as_default():
                    tf.summary.scalar("Average return", avg_return, step=step)
                    self.summary_writer.flush()
                returns.append(avg_return)

                # global_step = tf.compat.v1.train.get_global_step()

                self.train_checkpointer.save(global_step=step)

        # Save the policy at the end of training so that it can be easily deployed.
        # tf_policy_saver.save(policy_dir)
