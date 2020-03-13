from os.path import join

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __get_dataframe(self, filename: str):
        data = pd.read_csv(filename, header=None)
        data = data
        data.columns = ['camera_location', 'camera_rot_quat', 'target']
        data['target'] = data['target'].str.strip()

        def string_to_vec(string):
            return np.fromiter(map(lambda x: float(x[2:]), string.split()), dtype=float)

        data['camera_location'] = data['camera_location'].apply(string_to_vec)
        data['camera_rot_quat'] = data['camera_rot_quat'].apply(string_to_vec)

        X_location = np.array(data['camera_location'].tolist())
        X_rotation = np.array(data['camera_rot_quat'].tolist())
        X_total = np.concatenate([X_location, X_rotation], axis=1)

        self.standard_scaler = StandardScaler()
        X_total = self.standard_scaler.fit_transform(X_total)

        self.data, self.target = X_total, list(data['target'])

    def get_object_by_id(self, id: int):
        return self.data[id]

    def __init__(self, df_name: str, data_dir: str, batch_size=32, dim=(32, 32), n_channels=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.__get_dataframe(join(data_dir, df_name))
        self.list_IDs = list(range(self.data.shape[0]))
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return [X, np.ones(self.batch_size)], y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, 7))
        y = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.data[ID]

            # Store class
            y[i] = (img_to_array(load_img(join(self.data_dir, self.target[ID]), target_size=self.dim, interpolation='bicubic')) / 255).astype(np.float16)
        return X, y
