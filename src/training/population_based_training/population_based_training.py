import imp
import tensorflow as tf
from typing import List, Optional, Union
from tf_agents.agents import TFAgent
from agent.meta_agent.meta_agent import MetaAgent
from agent.meta_agent.meta_agent_copier import MetaAgentCopier
from summary_writer.summay_writer_manager_factory import SummaryWriterManagerFactory
from checkpoint_manager.agent_checkpoint_manager_factory import AgentCheckpointManagerFactory
from driver.driver_factory import DriverFactory
from agent.meta_agent.meta_q_agent.meta_q_agent import MetaQAgent
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
                 gradient_based_trainer: GradientBasedTraining,
                 fitness_evaluator: FitnessEvaluator,
                 agent_copier: MetaAgentCopier,
                 best_possible_fitness: Optional[int] = None
                 ) -> None:

        self.population = initial_population
        self.num_individuals = len(self.population)
        self.best = None

        self.gradient_based_trainer = gradient_based_trainer
        self.fitness_evaluator = fitness_evaluator
        self.agent_copier = agent_copier

        self.best_possible_fitness = best_possible_fitness

    def assess_fitness(self, meta_agent: MetaAgent) -> float:
        policy = meta_agent.tf_agent.policy
        fitness_value = self.fitness_evaluator.evaluate_fitness(policy=policy)

        return fitness_value

    def train(self):
        for meta_agent in self.population:
            meta_agent.previous_fitness = meta_agent.fitness

            self.gradient_based_trainer.train_agent(meta_agent=meta_agent)

            meta_agent.fitness = self.assess_fitness(meta_agent=meta_agent)

            if self.best is None or (meta_agent.fitness >= self.best.fitness):
                if self.best:
                    self.best.checkpoint_manager.delete_checkpointer()

                self.best = self.agent_copier.copy_agent(
                    meta_agent=meta_agent, name="best")

                if self.best_possible_fitness and self.best.fitness >= self.best_possible_fitness:
                    print("best_fitness: ", self.best.fitness)
                    return self.best

        print("best_fitness: ", self.best.fitness)

        return self.best
