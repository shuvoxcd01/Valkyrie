import tensorflow as tf
from Valkyrie.src.network.agent_network.network_factory import NetworkFactory
from Valkyrie.src.network.agent_network.q_networks.atari.atari_q_network import (
    AtariQNetwork,
)
from Valkyrie.src.network.pretraining_network.atari.atari_pretraining_network import (
    AtariPretrainingNetwork,
)
from tf_agents.specs import tensor_spec


class AtariQNetworkFactory(NetworkFactory):
    def __init__(
        self,
        pretraining_network: AtariPretrainingNetwork,
        input_tensor_spec,
        action_spec,
        conv_layer_params,
        fc_layer_params,
    ) -> None:
        super().__init__()

        self.action_tensor_spec = tensor_spec.from_spec(action_spec)
        self.num_actions = (
            self.action_tensor_spec.maximum - self.action_tensor_spec.minimum + 1
        )
        self.pretraining_network_encoder = pretraining_network.get_pretraining_network()
        self.input_tensor_spec = input_tensor_spec
        self.action_spec = action_spec
        self.conv_layer_params = conv_layer_params
        self.fc_layer_params = fc_layer_params

    def _create_dense_layer(self, num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode="fan_in", distribution="truncated_normal"
            ),
        )

    def get_network(self, kernel_initializer=None):
        q_values_layer = tf.keras.layers.Dense(
            self.num_actions,
            activation=None,
            kernel_initializer=kernel_initializer,
            bias_initializer=tf.keras.initializers.Constant(-0.2),
        )

        dense_layers = [
            self._create_dense_layer(num_units) for num_units in self.fc_layer_params
        ]

        trainable_layers = dense_layers + [q_values_layer]

        q_net = AtariQNetwork(
            pretraining_network=self.pretraining_network_encoder,
            layers=trainable_layers,
        )

        return q_net
