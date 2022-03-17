from tensorflow import keras
from configuration import config
import tensorflow as tf
import numpy as np


class Dataloader:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        self.x_train, self.y_train = self.preprocess(x_train, y_train)
        self.x_test, self.y_test = self.preprocess(x_test, y_test)
        self.digits = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    @staticmethod
    def preprocess(x, y):
        shape = (-1, 28, 28, 1) if config.dataset == 'mnist' else (-1, 32, 32, 3)
        x = (x.astype("float32") - 127.5) / 127.5
        x = np.reshape(x, shape)
        # y = keras.utils.to_categorical(y, 10)
        y = y.reshape(-1, )
        return x, y

    def get_dataset(self, task_idx=0, datatype='train', fake_label=None, multi_head=True):
        if datatype == 'train':
            data, label = self.x_train, self.y_train
        else:
            data, label = self.x_test, self.y_test
        # data_idx = np.isin(np.argmax(label, axis=1), self.digits[task_idx])
        data_idx = np.isin(label, self.digits[task_idx])
        data = data[data_idx]
        label = label[data_idx] if fake_label is None else np.ones_like(label[data_idx]) * fake_label
        if multi_head:
            # dataset = tf.data.Dataset.from_tensor_slices((data, label))
            return data, label - 2 * task_idx
        else:
            return data, label
            # dataset = tf.data.Dataset.from_tensor_slices(data)
        # dataset = dataset.shuffle(buffer_size=1024).batch(config['train']['batch_size'])
        # return dataset
