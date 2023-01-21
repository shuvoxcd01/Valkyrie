from Valkyrie.src.network.q_networks.cartpole.cartpole_q_network import CartPoleQNetwork
from network.network_factory import NetworkFactory
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.networks import sequential


class CartPoleQNetworkFactory(NetworkFactory):
    def __init__(self, input_tensor_spec, action_spec, conv_layer_params, fc_layer_params) -> None:
        super().__init__()

        self.fc_layer_params = (128, 256, 256, 128, 50)
        self.action_tensor_spec = tensor_spec.from_spec(action_spec)
        self.num_actions = self.action_tensor_spec.maximum - \
            self.action_tensor_spec.minimum + 1

    # Define a helper function to create Dense layers configured with the right
    # activation and kernel initializer.
    def dense_layer(self, num_units):
        return tf.keras.layers.Dense(
            num_units,
            activation=tf.keras.activations.relu,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0, mode='fan_in', distribution='truncated_normal'))

    # QNetwork consists of a sequence of Dense layers followed by a dense layer
    # with `num_actions` units to generate one q_value per available action as
    # its output.
    def get_network(self):
        dense_layers = [self.dense_layer(num_units)
                        for num_units in self.fc_layer_params]
        q_values_layer = tf.keras.layers.Dense(
            self.num_actions,
            activation=None,
            kernel_initializer=tf.keras.initializers.RandomUniform(
                minval=-0.03, maxval=0.03),
            bias_initializer=tf.keras.initializers.Constant(-0.2))
        q_net = CartPoleQNetwork(dense_layers + [q_values_layer])

        return q_net
