from datetime import datetime

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.experimental import CosineDecayRestarts
from tensorflow.keras.models import load_model

from data_generator import DataGenerator
from model_builder import build_base_generator, increase_generator_resolution

from tensorflow.keras.utils import plot_model
from ada_in import AdaInstanceNormalization

from draw_callback import DrawCallback

from tensorflow.keras.mixed_precision import experimental as mixed_precision
import argparse

def ms_ssim(true, pred):
    return 1 - tf.image.ssim_multiscale(true, pred, 1.0, power_factors=(0.5, 0.3, 0.2))


def configurate_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)



def set_everything_trainable(model: tf.keras.Model):
    for l in model.layers:
        l.trainable = True


def train_model(model: tf.keras.models.Model, epochs: int, directory: str, model_id: int = 0, initial_epoch: int = 0):
    decay = CosineDecayRestarts(0.0001, 15)
    optimizer = optimizers.Adam(learning_rate=decay, amsgrad=True)
    resolution = model.output_shape[1:3]

    if resolution[0] < 160:
        model.compile(optimizer=optimizer, loss='binary_crossentropy')
    else:
        model.compile(optimizer=optimizer, loss=ms_ssim)

    generator = DataGenerator(df_name='dataset.df', data_dir=directory, dim=resolution, n_channels=3,
                              batch_size=32,
                              shuffle=True)
    drawer = DrawCallback(model_id, generator.get_object_by_id(15009))
    check = ModelCheckpoint(filepath=f'style_nn_512x288_v4_{model_id}.h5')
    board = TensorBoard(log_dir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"), update_freq='epoch')
    model.fit(generator, callbacks=[drawer, check, board], workers=12, use_multiprocessing=True, epochs=epochs, initial_epoch = initial_epoch)


def progressive_learning(directory: str):
    new_model = build_base_generator(repeats=(3, 6, 6), channels=(256, 128, 128))
    plot_model(new_model, 'base_model.png', show_shapes=True)
    new_model.summary()
    train_model(new_model, epochs=100, directory=directory)

    addon_repeats = (6, 8, 8)
    addon_channels = (64, 64, 64)
    addon_epochs = (200, 300, 500)
    pretrain_new_layers = 10

    current_model = 1

    for repeats, channels, epochs in zip(addon_repeats, addon_channels, addon_epochs):
        new_model = increase_generator_resolution(new_model, repeats, channels, set_old_nontrainable=True)
        train_model(new_model, pretrain_new_layers, model_id=current_model)
        set_everything_trainable(new_model)
        train_model(new_model, epochs, directory=directory, model_id=current_model, initial_epoch = pretrain_new_layers)
        current_model += 1


def main(directory: str):
    configurate_gpu()
    progressive_learning(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('directory', type=str, help='Input files directory')
    args = parser.parse_args()
    main(args.directory)
