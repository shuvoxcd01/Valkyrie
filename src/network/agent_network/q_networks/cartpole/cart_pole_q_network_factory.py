import logging

import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.networks import sequential

from Valkyrie.src.network.agent_network.network_factory import NetworkFactory
from Valkyrie.src.network.agent_network.q_networks.cartpole.cartpole_q_network import (
    CartPoleQNetwork,
)
from Valkyrie.src.network.pretraining_network.cartpole.cartpole_pretraining_network import (
    CartPolePretrainingNetwork,
)


class CartPoleQNetworkFactory(NetworkFactory):
    def __init__(
        self,
        input_tensor_spec,
        action_spec,
        pretraining_network: CartPolePretrainingNetwork,
    ) -> None:
        super().__init__()

        self.action_tensor_spec = tensor_spec.from_spec(action_spec)
        self.num_actions = (
            self.action_tensor_spec.maximum - self.action_tensor_spec.minimum + 1
        )
        self.pretraining_network = pretraining_network

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def create_dense_layer(self, num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode="fan_in", distribution="truncated_normal"
            ),
        )

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    def get_network(self, kernel_initializer=None):
        q_values_layer = tf.keras.layers.Dense(
            self.num_actions,
            activation=None,
            kernel_initializer=kernel_initializer,
            bias_initializer=tf.keras.initializers.Constant(-0.2),
        )

        pretrained_layers = self.pretraining_network.get_pretrained_layers()
        trainable_dense_layer = self.create_dense_layer(100)
        trainable_layers = [trainable_dense_layer, q_values_layer]
        len_trainable_layers = len(trainable_layers)

        q_net = CartPoleQNetwork(
            pretrained_layers + trainable_layers,
            len_trainable_layers=len_trainable_layers,
        )

        return q_net
