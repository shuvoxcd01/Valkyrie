import tensorflow as tf
from tensorflow.keras.models import Model
from singleton_decorator import singleton


class BasePretrainingNetwork(Model):
    def __init__(self):
        super().__init__()
        self.encoder_network = None
        self.encoder_layers = None

    def get_pretrained_layers(self):
        self._check_build()

        self.encoder_network.trainable = False

        return self.encoder_layers

    def _check_build(self):
        if self.encoder_layers is None or self.encoder_network is None:
            raise Exception("Encoder layers have not been built.")

    def get_pretraining_network(self):
        self._check_build()

        self.encoder_network.trainable = False

        return self.encoder_network

    def get_encoder_output_spec(self):
        output_shape_without_batch = self.encoder_network.output_shape[1:]
        output_spec = tf.TensorSpec(shape=output_shape_without_batch)

        return output_spec
