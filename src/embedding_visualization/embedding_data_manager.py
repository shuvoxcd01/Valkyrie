import os
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
from tensorboard.plugins import projector


class EmbeddingDataManager:
    def __init__(self, base_metadata_dir) -> None:
        self.count = 0
        self.q_values = []
        self.tensors = []
        self.imgs = []
        self.save_dir = os.path.join(
            base_metadata_dir, "embedding_projector_metadata")
        self.img_dirname = "imgs"
        self.img_dir = os.path.join(self.save_dir, self.img_dirname)
        self.tensor_dirname = "tensors"
        self.tensor_dir = os.path.join(self.save_dir, self.tensor_dirname)
        self.metadata_filename = "metadata.tsv"
        self.sprite_filename = "sprite.png"
        self.sprite_single_img_dim = None
        self.sprite_path = os.path.join(self.save_dir, self.sprite_filename)
        self.metadata_filepath = os.path.join(
            self.save_dir, self.metadata_filename)

        self.initialize()

    def initialize(self):
        self.create_dirs_if_not_exist()
        self.create_metadata_file()

    def create_metadata_file(self):
        if not os.path.exists(self.metadata_filepath):
            df = pd.DataFrame(columns=["q_values"])
            df.to_csv(self.metadata_filepath, sep="\t", index=False)

    def create_dirs_if_not_exist(self):
        for _dir in [self.save_dir, self.img_dir, self.tensor_dir]:
            if not os.path.exists(_dir):
                os.makedirs(_dir)

    def save_img_individually(self, array: np.ndarray):
        filename = str(self.count)+".png"
        filepath = os.path.join(self.img_dir, filename)
        image = Image.fromarray(array)
        image.save(filepath)

    def save_tensor_individually(self, array: np.ndarray):
        filepath = os.path.join(self.tensor_dir, str(self.count)+".bytes")
        array.tofile(filepath)

    def save_metadata_individually(self, q_value: float):
        df = pd.read_table(self.metadata_filepath)
        assert len(df) == self.count
        df2 = pd.DataFrame(data={"q_values": [q_value]})
        df = pd.concat([df, df2], ignore_index=True)

        df.to_csv(self.metadata_filepath, sep='\t', index=False)

    def save(self):
        assert self.sprite_single_img_dim is None
        self.sprite_single_img_dim = [
            self.imgs[0].shape[2], self.imgs[0].shape[1]]

        df = pd.DataFrame(data=self.q_values)
        df.to_csv(self.metadata_filepath, sep='\t', index=False, header=False)

        #tensors = np.array(self.tensors)
        checkpoint = tf.train.Checkpoint(
            embedding=tf.Variable(tf.concat(self.tensors, 0)))
        checkpoint.save(os.path.join(self.save_dir, "embedding.ckpt"))
        # tensor_filepath = os.path.join(self.tensor_dir, "tensors.bytes")
        # tensors.tofile(tensor_filepath)

        img_to_sprite = np.concatenate(self.imgs, 0)
        sprite_img = self.images_to_sprite(img_to_sprite)
        sprite_img = Image.fromarray(sprite_img.astype(np.uint8))
        sprite_img.save(self.sprite_path)

    def add_data(self, img: np.ndarray, tensor: np.ndarray, q_value: float):
        self.imgs.append(np.expand_dims(img, 0))
        self.tensors.append(tensor)
        self.q_values.append(q_value)
        self.count += 1

    @staticmethod
    def images_to_sprite(data):
        if len(data.shape) == 3:
            data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
        data = data.astype(np.float32)
        min = np.min(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
        max = np.max(data.reshape((data.shape[0], -1)), axis=1)
        data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)

        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, 0),
                   (0, 0)) + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant',
                      constant_values=0)
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                               + tuple(range(4, data.ndim + 1)))
        data = data.reshape(
            (n * data.shape[1], n * data.shape[3]) + data.shape[4:])
        data = (data * 255).astype(np.uint8)
        return data

    def configure_projector(self):
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        # The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`.
        embedding.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
        embedding.metadata_path = self.metadata_filename
        embedding.sprite.image_path = self.sprite_filename
        assert self.sprite_single_img_dim is not None
        embedding.sprite.single_image_dim.extend(self.sprite_single_img_dim)
        projector.visualize_embeddings(self.save_dir, config)
