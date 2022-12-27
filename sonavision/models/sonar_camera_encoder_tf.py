import tensorflow as tf
from layers.unet_tf import CBRDownsample


def ENCODER(
    num_layers,
    input_shape=[256, 512, 3],
    base_filters=64,
):
    inp = tf.keras.layers.Input(input_shape)

    x = CBRDownsample(
            filters=base_filters, kernel_size=4, leakyReLU=True
        )(inp)
    for i in range(num_layers-1):
        x = CBRDownsample(
            filters=base_filters * (2 ** i+1), kernel_size=4, leakyReLU=True
        )(x)

    return tf.keras.Model(inputs=inp, outputs=x)