import tensorflow as tf

def CBR_downsample(filters=64, kernel_size=3, apply_batchnorm=True, leakyReLU=False):
    """Defines a downsampling Conv->Batch->ReLU Block.

    It is used iteratively when building Generator/Discriminator.
    Outputs a (1, 128, 128, 3) encoded image if given original dims of 256x256x3.

    Args:
        filters (int): number of filters
        size (int): kernel size
        apply_batchnorm (bool, optional): whether to apply batch norm. Defaults to True.
        leakyReLU (bool, optional): whether to apply leakyReLU or ReLU to final layer. Defaults to false

    Returns:
        tf.keras.Sequential: downsampling block
    """

    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    if leakyReLU:
        result.add(tf.keras.layers.LeakyReLU())
    else:
        result.add(tf.keras.layers.ReLU())

    return result


def CBR_upsample(filters=64, kernel_size=3, apply_dropout=False, leakyReLU=False):
    """Defines an upsamlping Conv->Batch->ReLU Block.

    It is used iteratively when building Generator / Discriminator.
    Outputs a (1, 256, 256, 3) decoded image given encoded dims of 128x128x3.

    Args:
        filters (int): number of filters
        size (int): kernel size
        apply_dropout (bool, optional): whether to apply dropout. Defaults to False.
        leakyReLU (bool, optional): whether to apply leakyReLU or ReLU to final layer. Defaults to false

    Returns:
        tf.keras.Sequential: upsampling block
    """
    initializer = tf.random_normal_initializer(0.0, 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    if leakyReLU:
        result.add(tf.keras.layers.LeakyReLU())
    else:
        result.add(tf.keras.layers.ReLU())

    return result
