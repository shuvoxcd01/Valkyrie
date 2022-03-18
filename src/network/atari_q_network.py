from tf_agents.networks import q_network
import tensorflow as tf


class AtariQNetwork(q_network.QNetwork):
    def call(self, observation, step_type=None, network_state=..., training=False):
        observation = tf.cast(observation, tf.float32)
        observation = observation / 255

        return super(AtariQNetwork, self).call(observation, step_type, network_state, training)
