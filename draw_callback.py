from tensorflow.keras.callbacks import Callback
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img


class DrawCallback(Callback):
    def __init__(self, model_id, object_to_predict):
        self.object_to_predict = object_to_predict
        self.model_id = model_id

    def __save(self, epoch):
        img = array_to_img(self.model.predict([np.array([self.object_to_predict]), np.array([1])])[0])
        img.save(f'./drawer/out_model_{self.model_id}_epoch_{epoch+1}.png')

    def on_train_begin(self, logs={}):
        self.__save(-1)

    def on_epoch_end(self, epoch, logs={}):
        self.__save(epoch)
