from tensorflow.keras.models import Model
from singleton_decorator import singleton


class BasePretrainingNetwork(Model):
    def __init__(self):
        super().__init__()
        self.encoder_network = None
        self.encoder_layers = None

    def get_pretrained_layers(self):
        if self.encoder_layers is None or self.encoder_network is None:
            raise Exception("Encoder layers are not built")

        self.encoder_network.trainable = False

        return self.encoder_layers
