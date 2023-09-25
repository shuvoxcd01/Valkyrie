from tensorflow.keras.models import Model
from singleton_decorator import singleton


# ToDo: Make the base class singleton so that all the child classes are singleton too.
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

    def train(self):
        self.encoder_network.trainable = True
        # ToDo: Change in trainable needs recompiling.
        self._train()
        self.encoder_network.trainable = False

    def _train(self):
        raise NotImplementedError()
