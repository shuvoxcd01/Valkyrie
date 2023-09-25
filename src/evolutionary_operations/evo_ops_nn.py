from typing import List
import tensorflow_probability as tfp
import logging

from Valkyrie.src.evolutionary_operations.evo_ops import EvolutionaryOperations


class EvolutionaryOperationsNN(EvolutionaryOperations):
    @staticmethod
    def crossover(
        parent_1_keep_precentage, parent_1_layers, parent_2_layers, child_layers
    ):
        for i in range(len(child_layers)):
            assert child_layers[
                i
            ].trainable, f"Network layer {child_layers[i]} is NOT trainable."

            partner_1_weights_list = parent_1_layers[i].get_weights()
            partner_2_weights_list = parent_2_layers[i].get_weights()

            assert len(partner_1_weights_list) == len(partner_2_weights_list)

            new_weights_list = []

            for j in range(len(partner_1_weights_list)):
                new_weights = (
                    parent_1_keep_precentage * partner_1_weights_list[j]
                    + (1 - parent_1_keep_precentage) * partner_2_weights_list[j]
                )

                new_weights_list.append(new_weights)

            child_layers[i].set_weights(new_weights_list)

    @staticmethod
    def mutate(mutation_layers: List, mean: float = 0.0, variance: float = 0.0001):
        d = tfp.distributions.Normal(loc=mean, scale=variance)

        for network_layer in mutation_layers:
            assert (
                network_layer.trainable
            ), f"Network layer {network_layer} is NOT trainable."

            weights_list = network_layer.get_weights()
            new_weights_list = EvolutionaryOperationsNN._get_tweaked_weights(
                d, weights_list
            )
            network_layer.set_weights(new_weights_list)

    @staticmethod
    def _get_tweaked_weights(d, weights_list):
        new_weights_list = []
        for weights in weights_list:
            noise = d.sample(sample_shape=weights.shape)
            new_weights = weights + noise
            new_weights_list.append(new_weights)
        return new_weights_list
