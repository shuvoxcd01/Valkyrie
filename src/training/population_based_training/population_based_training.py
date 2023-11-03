import random
import logging
from typing import List, Optional
from Valkyrie.src.training.pretraining.pretraining import Pretraining
from fitness_tracker.fitness_tracker import FitnessTracker
from parent_tracker.parent_tracker import ParentTracker
from agent.meta_agent.meta_agent import MetaAgent

from fitness_evaluator.fitness_evaluator import FitnessEvaluator
from training.gradient_based_training.gradient_based_training import (
    GradientBasedTraining,
)
import tensorflow as tf
from replay_buffer.replay_buffer_manager import ReplayBufferManager
import math
import numpy as np


class PopulationBasedTraining:
    def __init__(
        self,
        initial_population: List[MetaAgent],
        pretrainer: Pretraining,
        gradient_based_trainer: GradientBasedTraining,
        fitness_evaluator: FitnessEvaluator,
        fitness_trakcer: FitnessTracker,
        parent_tracker: ParentTracker,
        replay_buffer_manager: ReplayBufferManager,
        num_individuals: Optional[int] = None,
        best_possible_fitness: Optional[int] = None,
        num_training_iterations: int = 1000,
        generated_agent_prefix: str = "g_",
        generated_agent_count: int = 0,
        mutation_mean=0.0,
        mutation_variance=0.0001,
    ) -> None:
        self.population = initial_population
        self.num_individuals = (
            len(self.population) if num_individuals is None else num_individuals
        )

        self.best: MetaAgent = None

        self.gradient_based_trainer = gradient_based_trainer
        self.pretrainer = pretrainer
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
        self.replay_buffer_manager = replay_buffer_manager

    def assess_fitness(self, meta_agent: MetaAgent) -> float:
        policy = meta_agent.tf_agent.policy
        fitness_value = self.fitness_evaluator.evaluate_fitness(policy=policy)

        return fitness_value

    def _calculate_tweak_probability(self, meta_agent: MetaAgent):
        fitness_degradation = meta_agent.previous_fitness - meta_agent.fitness
        selective_tweak_prob = 1.0 / (1.0 + math.exp(-fitness_degradation))
        self.logger.debug(f"Selective tweak probability: {selective_tweak_prob}")

        random_tweak_prob = random.random()
        self.logger.debug(f"Random tweak probability: {random_tweak_prob}")

        tweak_prob = max(selective_tweak_prob, random_tweak_prob)
        self.logger.debug(f"Tweak probability: {tweak_prob}")

        return tweak_prob

    def train(self):
        for meta_agent in self.population:
            fitness = self.assess_fitness(meta_agent=meta_agent)
            meta_agent.update_fitness(fitness)

        for iteration in range(self.num_training_iterations):
            assert self.sanity_check(self.population), "Sanity check failed."
            self.pretrainer.train(agents_to_sync=self.population)

            for meta_agent in self.population:
                meta_agent.generation += 1

                meta_agent.previous_fitness = meta_agent.fitness

                self.fitness_tracker.write(
                    agent=meta_agent,
                    operation_name="None",
                    fitness_value=meta_agent.fitness,
                )

                self.gradient_based_trainer.train_agent(meta_agent=meta_agent)

                fitness_after_gradient_based_training = self.assess_and_update_fitness(
                    meta_agent, iteration=iteration, op="Gradient-based-training"
                )

                meta_agent.summary_writer_manager.write_scalar_summary(
                    "Fitness after gradient update",
                    data=fitness_after_gradient_based_training,
                    step=iteration,
                )

                if not self.best is None:
                    if (
                        self.best_possible_fitness
                        and self.best.fitness >= self.best_possible_fitness
                    ):
                        self.logger.info("best_fitness: ", self.best.fitness)
                        self.best.summary_writer_manager.write_scalar_summary(
                            "Best fitness",
                            data=meta_agent.fitness,
                            step=iteration,
                        )
                        return self.best

            population_sorted = sorted(
                self.population, key=lambda x: (x.fitness, x.generation), reverse=True
            )

            next_generation_population = population_sorted[: self.num_individuals]

            for parent_meta_agent in next_generation_population[:]:
                agent_name = self.generated_agent_prefix + str(
                    self.generated_agent_count
                )
                self.generated_agent_count += 1
                meta_agent = parent_meta_agent.copy(name=agent_name, generation=0)

                assert parent_meta_agent.tf_agent is not meta_agent.tf_agent
                assert (
                    parent_meta_agent.tf_agent._q_network
                    is not meta_agent.tf_agent._q_network
                )

                self.parent_tracker.write(
                    child_name=meta_agent.name,
                    parent_name=parent_meta_agent.name,
                    parent_generation=parent_meta_agent.generation,
                )

                self.logger.debug(f"Agent: {meta_agent.tf_agent.name}")

                self.perform_mutation(meta_agent, iteration, next_generation_population)

                self.perform_crossover(
                    meta_agent, iteration, next_generation_population
                )

            # for individual in self.population:
            #     if individual not in next_generation_population:
            #         individual.delete()

            self.population = next_generation_population
            population_names = [individual.name for individual in self.population]
            self.logger.info(f"Next generation population names: {population_names}")
            self.replay_buffer_manager.update_keep_only(population_names)
            self.logger.info("Replay buffer updated.")

        print("best_fitness: ", self.best.fitness)

        return self.best

    def perform_crossover(self, meta_agent, iteration, next_generation_population):
        (
            crossover_prob,
            soft_distance,
            soft_distance_idx,
        ) = self.crossover_probability(meta_agent, next_generation_population[:])

        if random.random() <= crossover_prob:
            crossover_partner = next_generation_population[:][soft_distance_idx]
            self.logger.debug("Performing crossover.")
            self.logger.debug(f"Parent 1: {meta_agent.tf_agent.name}")
            self.logger.debug(f"Parent 1 fitness: {meta_agent.fitness}")
            self.logger.debug(f"Parent 2 {crossover_partner.tf_agent.name}")
            self.logger.debug(f"Parent 2 fitness: {crossover_partner.fitness}")
            self.logger.debug(f"Parent 1 keep percentage: {crossover_prob}")

            self_keep_percentage = min(
                max(
                    (
                        1.0
                        / (
                            1.0
                            + math.exp(
                                -(meta_agent.fitness - crossover_partner.fitness)
                            )
                        )
                    ),
                    0.1,
                ),
                0.9,
            )

            child = meta_agent.crossover(
                partner=crossover_partner,
                self_keep_percentage=self_keep_percentage,
            )
            self.assess_and_update_fitness(child, iteration, op="Crossover")

            self.gradient_based_trainer.train_agent(child)

            self.assess_and_update_fitness(
                child, iteration, op="Gradient-based training after Crossover"
            )

            child.save()

            self.logger.debug("Crossover done.")
            self.logger.debug(f"Child fitness: {child.fitness}")
            child.summary_writer_manager.write_scalar_summary(
                "Fitness after crossover", data=child.fitness, step=iteration
            )

            if meta_agent in next_generation_population:
                next_generation_population.remove(meta_agent)

            next_generation_population.append(child)

        else:
            self.logger.debug("Skipping crossover.")

    def perform_mutation(self, meta_agent, iteration, next_generation_population):
        tweak_probability = self._calculate_tweak_probability(meta_agent)

        if random.random() <= tweak_probability:
            self.logger.debug("Starting mutation.")
            self.logger.debug(f"Agent fitness before mutation: {meta_agent.fitness}")

            meta_agent.mutate(mean=self.mutation_mean, variance=self.mutation_variance)

            fitness_after_mutation = self.assess_and_update_fitness(
                meta_agent, iteration, op="Mutation"
            )

            self.gradient_based_trainer.train_agent(meta_agent)

            fitness_after_mutation = self.assess_and_update_fitness(
                meta_agent,
                iteration,
                op="Gradient-based training after Mutation",
            )

            self.logger.debug("Mutation Finished.")
            self.logger.debug(f"Agent fitness after mutation: {meta_agent.fitness}")

            meta_agent.summary_writer_manager.write_scalar_summary(
                "Fitness after mutation",
                data=fitness_after_mutation,
                step=iteration,
            )

            next_generation_population.append(meta_agent)

        else:
            self.logger.debug("Skipping mutation.")

    def assess_and_update_fitness(self, meta_agent, iteration, op: str = "None"):
        fitness_after_gradient_based_training = self.assess_fitness(
            meta_agent=meta_agent
        )

        meta_agent.update_fitness(fitness_after_gradient_based_training)

        self.fitness_tracker.write(
            agent=meta_agent,
            operation_name=f"Operation: {op}",
            fitness_value=meta_agent.fitness,
        )

        self.check_and_update_best(meta_agent, iteration)

        return fitness_after_gradient_based_training

    def check_and_update_best(self, meta_agent, iteration):
        if self.best is None or (meta_agent.fitness >= self.best.fitness):
            # ToDo: Fix checkpointer
            if self.best:
                self.best.delete()

            self.best = meta_agent.copy(name="best")
            self.best.save()

            self.best.summary_writer_manager.write_scalar_summary(
                "Best fitness",
                data=meta_agent.fitness,
                step=iteration,
            )

    def _calculate_beta(self, meta_agent: MetaAgent):
        beta = 0

        minimum_required_fitness = self.best.fitness / 2.0

        if self.best.fitness < 0:
            minimum_required_fitness = self.best.fitness * 2

        if (
            meta_agent.fitness < minimum_required_fitness
            and meta_agent.previous_fitness < minimum_required_fitness
        ):
            beta = 0.5

        return beta

    def _calculate_distance(self, layers_1: List, layers_2: List):
        assert len(layers_1) == len(layers_2), "Layers don't have same length."

        for layer_1, layer_2 in zip(layers_1, layers_2):
            _sum = 0.0
            for layer_1_weights, layer_2_weights in zip(
                layer_1.get_weights(), layer_2.get_weights()
            ):
                _sum += tf.reduce_sum(tf.abs(layer_1_weights - layer_2_weights)).numpy()

        return _sum

    def calculate_soft_distance(self, values: List):
        values = tf.convert_to_tensor(values, dtype=tf.float32)
        soft_distance = tf.nn.softmax(values).numpy()

        return soft_distance

    def crossover_probability(self, individual: MetaAgent, candidates: List[MetaAgent]):
        crossover_prob = 0.0

        # ToDo: Remove crossover with self.
        # if individual in candidates:
        #     candidates.remove(individual)

        individual_network = individual.get_network()
        individual_crossover_layers = individual_network.get_crossover_layers()

        distances = []

        for candidate in candidates:
            candidate_network = candidate.get_network()
            candidate_crossover_layers = candidate_network.get_crossover_layers()

            distances.append(
                self._calculate_distance(
                    individual_crossover_layers, candidate_crossover_layers
                )
            )

        soft_distances = self.calculate_soft_distance(distances)

        soft_distance = np.max(soft_distances)
        self.logger.info(f"Soft distance: {soft_distance}")

        soft_distance_index = np.argmax(soft_distances)
        self.logger.info(f"Soft distance index: {soft_distance_index}")

        crossover_prob = max(soft_distance, random.random())
        self.logger.info(f"Crossover probability: {crossover_prob}")

        return crossover_prob, soft_distance, soft_distance_index

    def sanity_check(self, agents: List[MetaAgent]):
        if len(agents) <= 1:
            return True

        ground_agent = agents[0]
        ground_agent_layers = (
            ground_agent.get_network().get_pretraining_network().layers
        )

        for agent in agents[1:]:
            agent_layers = agent.get_network().get_pretraining_network().layers

            _sum = 0.0

            for ground_layer, agent_layer in zip(ground_agent_layers, agent_layers):
                ground_layer_weights_list = ground_layer.get_weights()
                agent_layer_weights_list = agent_layer.get_weights()

                for gl_weights, al_weights in zip(
                    ground_layer_weights_list, agent_layer_weights_list
                ):
                    _sum += tf.reduce_sum(gl_weights - al_weights).numpy()

            return _sum == 0.0
