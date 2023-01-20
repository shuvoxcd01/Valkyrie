from typing import List
import tensorflow_probability as tfp
import logging

from Valkyrie.src.evolutionary_operations.evo_ops import EvolutionaryOperations


class EvolutionaryOperationsNN(EvolutionaryOperations):
    @staticmethod
    def crossover(parent_1_keep_precentage, parent_network_1, parent_network_2, child_network,
                  crossover_layer_indices: List = None):
        if crossover_layer_indices is None:
            logging.info("Performing full crossover.")
            crossover_layer_indices = range(len(child_network.layers))
        else:
            logging.info(
                f"Performing crossover on specific indices: {crossover_layer_indices}.")
            for i in crossover_layer_indices:
                assert child_network.layers[
                    i].trainable, f"Network layer {child_network.layers[i]} is NOT trainable."

        for i in crossover_layer_indices:
            if child_network.layers[i].trainable:
                partner_1_weights_list = parent_network_1.layers[i].get_weights(
                )
                partner_2_weights_list = parent_network_2.layers[i].get_weights(
                )

                assert len(partner_1_weights_list) == len(
                    partner_2_weights_list)

                new_weights_list = []

                for j in range(len(partner_1_weights_list)):
                    new_weights = parent_1_keep_precentage * \
                        partner_1_weights_list[j] + \
                        (1-parent_1_keep_precentage)*partner_2_weights_list[j]

                    new_weights_list.append(new_weights)

                child_network.layers[i].set_weights(new_weights_list)

    @staticmethod
    def mutate(network, mut_layer_indices: List, mean: float = 0., variance: float = 0.0001):
        d = tfp.distributions.Normal(loc=mean, scale=variance)

        for index in mut_layer_indices:
            network_layer = network.layers[index]
            assert network_layer.trainable, f"Network layer {network_layer} is NOT trainable."

            weights_list = network_layer.get_weights()
            new_weights_list = EvolutionaryOperationsNN._get_tweaked_weights(
                d, weights_list)
            network_layer.set_weights(new_weights_list)

    @staticmethod
    def _get_tweaked_weights(d, weights_list):
        new_weights_list = []
        for weights in weights_list:
            noise = d.sample(sample_shape=weights.shape)
            new_weights = weights + noise
            new_weights_list.append(new_weights)
        return new_weights_list
