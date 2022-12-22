import tensorflow as tf
from layers.unet_tf import CBRDownsample, CBRUpsample


def PIX2PIX_GENERATOR_SONAR_CAMERA(
    input_shape=[256, 512, 3],
    output_shape=[256, 512, 3],
    base_filters=64,
):
    """Generator for the pix2pix model.

    Args:
        input_shape (list, optional): shape of the input vector. Defaults to [256, 512, 3].
        output_shape (list, optional): shape of the output vector. Defaults to [256, 512, 3].
        base_filters (int, optional): Number of base filters. Defaults to 64.

    Raises:
        ValueError: If the input shape is not power of 2.

    Returns:
        _type_: tf.keras.Model
    """
    sonar_input = tf.keras.layers.Input(input_shape)
    camera_input = tf.keras.layers.Input(input_shape)

    sonar_input_conv = tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding="same"
    )(sonar_input)
    camera_input_conv = tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding="same"
    )(camera_input)

    concat = tf.keras.layers.Concatenate()([sonar_input_conv, camera_input_conv])

    min_dim = min(input_shape[0], input_shape[1])
    temp_min_dim = min_dim

    # check which power of 2 is the min_dim
    count = 0
    while temp_min_dim > 1:
        count += 1
        temp_min_dim = temp_min_dim / 2

    if temp_min_dim != 1:
        raise ValueError("The min dimension of the input shape must be a power of 2")

    # create downsample stack of layers for Yx1 resolution
    downsample_stack = []
    for i in range(
        1, count - 2
    ):  # may be reduce this to keep trainable params in check
        downsample_stack.append(
            CBRDownsample(
                filters=base_filters * (2 ** i), kernel_size=4, leakyReLU=True
            )
        )

    # create upsample stack of layers for coming back to output resolution
    upsample_stack = []
    for i in range(
        1, count - 2
    ):  # may be reduce this to keep trainable params in check
        upsample_stack.append(
            CBRUpsample(filters=base_filters * (2 ** i), kernel_size=4, leakyReLU=True)
        )

    # create output layer
    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        output_shape[2],
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )

    # skip connections
    skip_connections = []
    for d in downsample_stack:
        concat = d(concat)
        skip_connections.append(concat)

    skip_connections = reversed(skip_connections[:-1])

    for up, skip in zip(upsample_stack, skip_connections):
        concat = up(concat)
        concat = tf.keras.layers.Concatenate()([concat, skip])

    output = last(concat)

    return tf.keras.Model(inputs=[sonar_input, camera_input], outputs=output)


def PIX2PIX_DISCRIMINATOR_SONAR_CAMERA(
    input_shape=[256, 512, 3],
    output_shape=[256, 512, 3],
    base_filters=256,
):
    # define PatchGAN discriminator
    sonar_input = tf.keras.layers.Input(input_shape)
    camera_input = tf.keras.layers.Input(input_shape)
    reference_input = tf.keras.layers.Input(input_shape)

    sonar_input_conv = tf.keras.layers.Conv2D(
        filters=256, kernel_size=3, strides=1, padding="same"
    )(sonar_input)
    camera_input_conv = tf.keras.layers.Conv2D(
        filters=256, kernel_size=3, strides=1, padding="same"
    )(camera_input)

    # TODO: how to concatenate the inputs?
    concat = tf.keras.layers.Concatenate()(
        [sonar_input_conv, camera_input_conv, reference_input]
    )

    # TODO: how to go down till 70x140 what is equivalent?
    d1 = CBRDownsample(
        filters=base_filters, kernel_size=4, apply_batchnorm=False, leakyReLU=True
    )(concat)
    d2 = CBRDownsample(filters=base_filters * 2, kernel_size=4, leakyReLU=True)(d1)
    d3 = CBRDownsample(filters=base_filters * 4, kernel_size=4, leakyReLU=True)(d2)
    d4 = CBRDownsample(filters=base_filters * 8, kernel_size=4, leakyReLU=True)(d3)
    d5 = CBRDownsample(filters=1, kernel_size=4, leakyReLU=True)(d4)

    return tf.keras.Model(
        inputs=[sonar_input, camera_input, reference_input], outputs=d5
    )

def PIX2PIX_GENERATOR(
    input_shape=[256, 512, 3],
    output_shape=[256, 512, 3],
    base_filters=64,
):
    """Generator for the pix2pix model.

    Args:
        input_shape (list, optional): shape of the input vector. Defaults to [256, 512, 3].
        output_shape (list, optional): shape of the output vector. Defaults to [256, 512, 3].
        base_filters (int, optional): Number of base filters. Defaults to 64.

    Raises:
        ValueError: If the input shape is not power of 2.

    Returns:
        _type_: tf.keras.Model
    """
    # sonar_input = tf.keras.layers.Input(input_shape)
    camera_input = tf.keras.layers.Input(input_shape)

    # sonar_input_conv = tf.keras.layers.Conv2D(
    #     filters=16, kernel_size=3, strides=1, padding="same"
    # )(sonar_input)
    camera_input_conv = tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding="same"
    )(camera_input)

    concat = camera_input_conv

    min_dim = min(input_shape[0], input_shape[1])
    temp_min_dim = min_dim

    # check which power of 2 is the min_dim
    count = 0
    while temp_min_dim > 1:
        count += 1
        temp_min_dim = temp_min_dim / 2

    if temp_min_dim != 1:
        raise ValueError("The min dimension of the input shape must be a power of 2")

    # create downsample stack of layers for Yx1 resolution
    downsample_stack = []
    for i in range(
        1, count - 2
    ):  # may be reduce this to keep trainable params in check
        downsample_stack.append(
            CBRDownsample(
                filters=base_filters * (2 ** i), kernel_size=4, leakyReLU=True
            )
        )

    # create upsample stack of layers for coming back to output resolution
    upsample_stack = []
    for i in range(
        1, count - 2
    ):  # may be reduce this to keep trainable params in check
        upsample_stack.append(
            CBRUpsample(filters=base_filters * (2 ** i), kernel_size=4, leakyReLU=True)
        )

    # create output layer
    initializer = tf.random_normal_initializer(0.0, 0.02)
    last = tf.keras.layers.Conv2DTranspose(
        output_shape[2],
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )

    # skip connections
    skip_connections = []
    for d in downsample_stack:
        concat = d(concat)
        skip_connections.append(concat)

    skip_connections = reversed(skip_connections[:-1])

    for up, skip in zip(upsample_stack, skip_connections):
        concat = up(concat)
        concat = tf.keras.layers.Concatenate()([concat, skip])

    output = last(concat)

    return tf.keras.Model(inputs= camera_input, outputs=output)


def PIX2PIX_DISCRIMINATOR(
    input_shape=[256, 512, 3],
    output_shape=[256, 512, 3],
    base_filters=256,
):
    # define PatchGAN discriminator
    # sonar_input = tf.keras.layers.Input(input_shape)
    camera_input = tf.keras.layers.Input(input_shape)
    reference_input = tf.keras.layers.Input(input_shape)

    # sonar_input_conv = tf.keras.layers.Conv2D(
    #     filters=256, kernel_size=3, strides=1, padding="same"
    # )(sonar_input)
    camera_input_conv = tf.keras.layers.Conv2D(
        filters=256, kernel_size=3, strides=1, padding="same"
    )(camera_input)

    # TODO: how to concatenate the inputs?
    concat = tf.keras.layers.Concatenate()(
        [camera_input_conv, reference_input]
    )

    # TODO: how to go down till 70x140 what is equivalent?
    d1 = CBRDownsample(
        filters=base_filters, kernel_size=4, apply_batchnorm=False, leakyReLU=True
    )(concat)
    d2 = CBRDownsample(filters=base_filters * 2, kernel_size=4, leakyReLU=True)(d1)
    d3 = CBRDownsample(filters=base_filters * 4, kernel_size=4, leakyReLU=True)(d2)
    d4 = CBRDownsample(filters=base_filters * 8, kernel_size=4, leakyReLU=True)(d3)
    d5 = CBRDownsample(filters=1, kernel_size=4, leakyReLU=True)(d4)

    return tf.keras.Model(
        inputs=[camera_input, reference_input], outputs=d5
    )
