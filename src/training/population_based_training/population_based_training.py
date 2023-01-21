import random
import logging
from typing import List, Optional
from fitness_tracker.fitness_tracker import FitnessTracker
from parent_tracker.parent_tracker import ParentTracker
from agent.meta_agent.meta_agent import MetaAgent

from fitness_evaluator.fitness_evaluator import FitnessEvaluator
from training.gradient_based_training.gradient_based_training import GradientBasedTraining
import math


class PopulationBasedTraining:
    def __init__(self, initial_population: List[MetaAgent],
                 gradient_based_trainer: GradientBasedTraining,
                 fitness_evaluator: FitnessEvaluator,
                 fitness_trakcer: FitnessTracker,
                 parent_tracker: ParentTracker,
                 num_individuals: Optional[int] = None,
                 best_possible_fitness: Optional[int] = None,
                 num_training_iterations: int = 100,
                 generated_agent_prefix: str = "g_",
                 generated_agent_count: int = 0,
                 mutation_mean=0.0,
                 mutation_variance=0.0001
                 ) -> None:

        self.population = initial_population
        self.num_individuals = len(
            self.population) if num_individuals is None else num_individuals

        self.best = None

        self.gradient_based_trainer = gradient_based_trainer
        self.fitness_evaluator = fitness_evaluator

        self.best_possible_fitness = best_possible_fitness
        self.logger = logging.getLogger()
        self.num_training_iterations = num_training_iterations
        self.fitness_tracker = fitness_trakcer
        self.parent_tracker = parent_tracker
        self.generated_agent_prefix = generated_agent_prefix
        self.generated_agent_count = generated_agent_count
        self.mutation_mean = mutation_mean
        self.mutation_variance = mutation_variance

    def assess_fitness(self, meta_agent: MetaAgent) -> float:
        policy = meta_agent.tf_agent.policy
        fitness_value = self.fitness_evaluator.evaluate_fitness(policy=policy)

        return fitness_value

    @staticmethod
    def _calculate_tweak_probability(meta_agent: MetaAgent):
        fitness_degradation = meta_agent.previous_fitness - meta_agent.fitness
        tweak_prob = 1./(1. + math.exp(-fitness_degradation))

        return tweak_prob

    def train(self):
        for meta_agent in self.population:
            fitness = self.assess_fitness(meta_agent=meta_agent)
            meta_agent.update_fitness(fitness)

        for iteration in range(self.num_training_iterations):
            for meta_agent in self.population:
                meta_agent.generation += 1

                meta_agent.previous_fitness = meta_agent.fitness

                self.fitness_tracker.write(
                    agent=meta_agent, operation_name="None", fitness_value=meta_agent.fitness)

                self.gradient_based_trainer.train_agent(meta_agent=meta_agent)

                fitness_after_gradient_based_training = self.assess_fitness(
                    meta_agent=meta_agent)

                meta_agent.update_fitness(
                    fitness_after_gradient_based_training)

                self.fitness_tracker.write(
                    agent=meta_agent, operation_name="Gradient-based-training", fitness_value=meta_agent.fitness)

                meta_agent.summary_writer_manager.write_scalar_summary(
                    "Fitness after gradient update", data=fitness_after_gradient_based_training, step=iteration)

                if self.best is None or (meta_agent.fitness >= self.best.fitness):
                    if self.best:
                        self.best.checkpoint_manager.delete_checkpointer()

                    self.best = meta_agent.copy(name="best")

                    self.best.summary_writer_manager.write_scalar_summary(
                        "Best fitness", data=fitness_after_gradient_based_training, step=iteration)

                    if self.best_possible_fitness and self.best.fitness >= self.best_possible_fitness:
                        print("best_fitness: ", self.best.fitness)
                        return self.best

            population_sorted = sorted(
                self.population, key=lambda x: (x.fitness, x.generation), reverse=True)

            next_generation_population = population_sorted[:self.num_individuals]

            for parent_meta_agent in next_generation_population[:]:
                agent_name = self.generated_agent_prefix + \
                    str(self.generated_agent_count)
                self.generated_agent_count += 1
                meta_agent = parent_meta_agent.copy(
                    name=agent_name, generation=0)

                assert parent_meta_agent.tf_agent is not meta_agent.tf_agent
                assert parent_meta_agent.tf_agent._q_network is not meta_agent.tf_agent._q_network

                self.parent_tracker.write(
                    child_name=meta_agent.name, parent_name=parent_meta_agent.name,
                    parent_generation=parent_meta_agent.generation)

                self.logger.debug(f"Agent: {meta_agent.tf_agent.name}")

                tweak_probability = self._calculate_tweak_probability(
                    meta_agent)

                if random.random() >= tweak_probability:
                    self.logger.debug("Starting mutation.")
                    self.logger.debug(
                        f"Agent fitness before mutation: {meta_agent.fitness}")

                    meta_agent.mutate(mean=self.mutation_mean,
                                      variance=self.mutation_variance)

                    fitness_after_mutation = self.assess_fitness(
                        meta_agent=meta_agent)
                    meta_agent.update_fitness(fitness_after_mutation)

                    self.logger.debug("Mutation Finished.")
                    self.logger.debug(
                        f"Agent fitness after mutation: {meta_agent.fitness}")

                    meta_agent.summary_writer_manager.write_scalar_summary(
                        "Fitness after mutation", data=fitness_after_mutation, step=iteration)

                    self.fitness_tracker.write(
                        agent=meta_agent, operation_name="Mutation", fitness_value=meta_agent.fitness)

                    next_generation_population.append(meta_agent)

                else:
                    self.logger.debug("Skipping mutation.")

                beta = self._calculate_beta(meta_agent)

                if beta:
                    self.logger.debug("Performing crossover.")
                    self.logger.debug(f"Parent 1: {meta_agent.tf_agent.name}")
                    self.logger.debug(
                        f"Parent 1 fitness: {meta_agent.fitness}")
                    self.logger.debug(f"Parent 2 {self.best.tf_agent.name}")
                    self.logger.debug(f"Parent 2 fitness: {self.best.fitness}")
                    self.logger.debug(
                        f"Parent 1 keep percentage: {beta}")

                    child = meta_agent.crossover(
                        partner=self.best, self_keep_percentage=beta)
                    child.update_fitness(self.fitness_evaluator.evaluate_fitness(
                        policy=child.tf_agent.policy))
                    child.save()

                    self.logger.debug("Crossover done.")
                    self.logger.debug(f"Child fitness: {child.fitness}")
                    child.summary_writer_manager.write_scalar_summary(
                        "Fitness after crossover", data=child.fitness, step=iteration)

                    self.fitness_tracker.write(
                        agent=child, operation_name="Crossover", fitness_value=child.fitness)

                    if meta_agent in next_generation_population:
                        next_generation_population.remove(meta_agent)
                    meta_agent.checkpoint_manager.delete_checkpointer()
                    next_generation_population.append(child)

                else:
                    self.logger.debug("Skipping crossover.")

            self.population = next_generation_population
        print("best_fitness: ", self.best.fitness)

        return self.best

    def _calculate_beta(self, meta_agent: MetaAgent):
        beta = 0

        minimum_required_fitness = self.best.fitness / 2.

        if self.best.fitness < 0:
            minimum_required_fitness = self.best.fitness * 2

        if meta_agent.fitness < minimum_required_fitness and meta_agent.previous_fitness < minimum_required_fitness:
            beta = 0.5

        return beta
