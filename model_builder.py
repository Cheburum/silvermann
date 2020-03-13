from tensorflow.keras.layers import Input, Dense, Conv2DTranspose, Reshape, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model
from tensorflow import Tensor

from ada_in import AdaInstanceNormalization


def __latent_block(input_layer: Tensor, n_layers: int = 2, units: int = 256, activation: str = 'relu'):
    name = 'style_0' if n_layers > 1 else 'style'
    x = Dense(units, activation=activation, name=name, trainable=True)(input_layer)
    for i in range(n_layers - 1):
        name = f'style_{i + 1}' if i < n_layers - 2 else 'style'
        x = Dense(units, activation=activation, name=name, trainable=True)(x)
    return x


def __ada_inst(style: Tensor, x: Tensor):
    style1_b = Dense(x.shape[-1], activation='relu', trainable=True)(style)
    style1_b = Reshape((1, 1, x.shape[-1]))(style1_b)
    style1_g = Dense(x.shape[-1], activation='relu', trainable=True)(style)
    style1_g = Reshape((1, 1, x.shape[-1]))(style1_g)
    x = AdaInstanceNormalization()([x, style1_b, style1_g])
    return x


def __res_block(x: Tensor, batch_norm: bool = True, kernel_size: (int, int) = (3, 3), channels: int = 64,
                block_size: int = 3):
    assert (block_size >= 3)
    x_shortcut = None
    for i in range(block_size - 1):
        x = Conv2DTranspose(channels, kernel_size, strides=(1, 1), padding='same', trainable=True)(x)
        if batch_norm:
            x = BatchNormalization(trainable=True)(x)
        x = Activation('relu')(x)
        if i == 0:
            x_shortcut = x
        elif i == block_size - 2:
            x = Add()([x_shortcut, x])
    return x


def _shortcut(x_shortcut: Tensor, x: Tensor):
    input_shape = x.shape
    residual_shape = x_shortcut.shape
    print(input_shape)
    stride_width = int(round(input_shape[1] / residual_shape[1]))
    stride_height = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[-1] == residual_shape[-1]

    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        x_shortcut = Conv2DTranspose(filters=input_shape[-1],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="same",
                          kernel_initializer="he_normal")(x_shortcut)

    return Add()([x_shortcut, x])

def __res_bottleneck_block(x: Tensor, batch_norm: bool = True, kernel_size: (int, int) = (3, 3), channels: int = 64):
    x_shortcut = x

    if batch_norm:
        x = BatchNormalization(trainable=True)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(channels * 4, (1, 1), strides=(1, 1), padding='same', trainable=True)(x)

    if batch_norm:
        x = BatchNormalization(trainable=True)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(channels, kernel_size, strides=(1, 1), padding='same', trainable=True)(x)

    if batch_norm:
        x = BatchNormalization(trainable=True)(x)
    x = Activation('relu')(x)
    x = Conv2DTranspose(channels, (1, 1), strides=(1, 1), padding='same', trainable=True)(x)

    x = _shortcut(x_shortcut, x)
    return x


def __up_conv_block(x: Tensor, batch_norm: bool = True, kernel_size: tuple = (3, 3), channels: int = 64):
    x = Conv2DTranspose(channels, kernel_size, strides=(2, 2), padding='same', trainable=True)(x)
    if batch_norm:
        x = BatchNormalization(trainable=True)(x)
    x = Activation('relu')(x)
    return x


def __constant_image(size: (int, int, int) = (9, 16, 128)) -> (Tensor, Tensor):
    constant_image = Input(shape=[1], name='constant_input')
    x = Dense(size[0] * size[1] * size[2], kernel_initializer='he_normal', trainable=True)(constant_image)
    x = Reshape(size)(x)
    return constant_image, x


def build_base_generator(repeats: tuple = (4, 4, 6, 8, 8), channels: tuple = (128, 128, 64, 64, 32),
                         block_size: int = 4) -> Model:
    input_vec = Input(shape=(7,), name='input')
    style = __latent_block(input_vec, 2, 256, 'relu')
    input_constant, current_layer = __constant_image((9, 16, channels[0]))
    current_layer = __ada_inst(style, current_layer)
    current_layer = __res_bottleneck_block(current_layer, channels=channels[0])

    for i, j in zip(repeats, channels):
        current_layer = __up_conv_block(current_layer, channels=j)
        current_layer = __ada_inst(style, current_layer)
        for _ in range(i):
            current_layer = __res_bottleneck_block(current_layer, channels=j)
    output = Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', name='output',
                             trainable=True)(
        current_layer)
    output = Activation('sigmoid', dtype='float32')(output)
    neural_render = Model([input_vec, input_constant], output)
    return neural_render


def increase_generator_resolution(neural_render: Model, repeat: int, channels: int, set_old_nontrainable: bool,
                                  block_size: int = 4) -> Model:
    if set_old_nontrainable:
        for l in neural_render.layers:
            l.trainable = False
    style = neural_render.get_layer(name='style').output
    current_layer = neural_render.get_layer(index=-2).output
    current_layer = __up_conv_block(current_layer, channels=channels)
    current_layer = __ada_inst(style, current_layer)
    current_layer = __res_bottleneck_block(current_layer, channels=channels)
    output = Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', name='output')(
        current_layer)
    output = Activation('sigmoid', dtype='float32')(output)
    input_vec = neural_render.get_layer(name='input').output
    input_constant = neural_render.get_layer(name='constant_input').output
    neural_render = Model([input_vec, input_constant], output)
    return neural_render
