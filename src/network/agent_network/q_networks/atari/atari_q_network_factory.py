

from Valkyrie.src.network.agent_network.network_factory import NetworkFactory
from Valkyrie.src.network.agent_network.q_networks.atari.atari_q_network import AtariQNetwork


class AtariQNetworkFactory(NetworkFactory):
    def __init__(self, input_tensor_spec, action_spec, conv_layer_params, fc_layer_params) -> None:
        super().__init__()

        self.input_tensor_spec = input_tensor_spec
        self. action_spec = action_spec
        self.conv_layer_params = conv_layer_params
        self.fc_layer_params = fc_layer_params

    def get_network(self, kernel_initializer=None):
        q_net = AtariQNetwork(
            input_tensor_spec=self.input_tensor_spec,
            action_spec=self.action_spec,
            conv_layer_params=self.conv_layer_params,
            fc_layer_params=self.fc_layer_params,
            kernel_initializer=kernel_initializer
        )

        return q_net