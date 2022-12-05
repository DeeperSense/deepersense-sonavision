import tensorflow as tf
from layers.unet_tf import CBRDownsample, CBRUpsample


def PIX2PIX_GENERATOR_SONAR_CAMERA(
    input_shape=[256, 512, 3],
    output_shape=[256, 512, 3],
):
    sonar_input = tf.keras.layers.Input(input_shape)
    camera_input = tf.keras.layers.Input(input_shape)

    sonar_input_conv = tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding="same"
    )(sonar_input)
    camera_input_conv = tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding="same"
    )(camera_input)

    
    max_dim = max(input_shape[0], input_shape[1])
    min_dim = min(input_shape[0], input_shape[1])
    max_dim_index = input_shape.index(max_dim)
    min_dim_index = input_shape.index(min_dim)

    temp_min_dim = min_dim

    # TODO: create downsample stack of layers for Yx1 resolution

    # TODO: create upsample stack of layers for coming back to output resolution


def PIX2PIX_DISCRIMINATOR_SONAR_CAMERA(
    input_shape=[256, 512, 3],
):
    pass
