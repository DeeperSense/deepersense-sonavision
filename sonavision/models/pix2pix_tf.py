import tensorflow as tf
from layers.unet_tf import CBRDownsample, CBRUpsample


# def PIX2PIX_GENERATOR_SONAR_CAMERA(
#     arch_type,
#     input_shape=[256, 512, 3],
#     output_shape=[256, 512, 3],
#     base_filters=64,
# ):
#     """Generator for the pix2pix model.

#     Args:
#         input_shape (list, optional): shape of the input vector. Defaults to [256, 512, 3].
#         output_shape (list, optional): shape of the output vector. Defaults to [256, 512, 3].
#         base_filters (int, optional): Number of base filters. Defaults to 64.

#     Raises:
#         ValueError: If the input shape is not power of 2.

#     Returns:
#         _type_: tf.keras.Model
#     """
#     sonar_input = tf.keras.layers.Input(input_shape)
#     camera_input = tf.keras.layers.Input(input_shape)

#     sonar_input_conv = tf.keras.layers.Conv2D(
#         filters=16, kernel_size=3, strides=1, padding="same"
#     )(sonar_input)
#     camera_input_conv = tf.keras.layers.Conv2D(
#         filters=16, kernel_size=3, strides=1, padding="same"
#     )(camera_input)

#     concat = tf.keras.layers.Concatenate()([sonar_input_conv, camera_input_conv])

#     min_dim = min(input_shape[0], input_shape[1])
#     temp_min_dim = min_dim

#     # check which power of 2 is the min_dim
#     count = 0
#     while temp_min_dim > 1:
#         count += 1
#         temp_min_dim = temp_min_dim / 2

#     if temp_min_dim != 1:
#         raise ValueError("The min dimension of the input shape must be a power of 2")

#     # create downsample stack of layers for Yx1 resolution
#     downsample_stack = []
#     for i in range(
#         1, count - 2
#     ):  # may be reduce this to keep trainable params in check
#         downsample_stack.append(
#             CBRDownsample(
#                 filters=base_filters * (2 ** i), kernel_size=4, leakyReLU=True
#             )
#         )

#     # create upsample stack of layers for coming back to output resolution
#     upsample_stack = []
#     for i in range(
#         1, count - 2
#     ):  # may be reduce this to keep trainable params in check
#         upsample_stack.append(
#             CBRUpsample(filters=base_filters * (2 ** i), kernel_size=4, leakyReLU=True)
#         )

#     # create output layer
#     initializer = tf.random_normal_initializer(0.0, 0.02)
#     last = tf.keras.layers.Conv2DTranspose(
#         output_shape[2],
#         4,
#         strides=2,
#         padding="same",
#         kernel_initializer=initializer,
#         activation="tanh",
#     )

#     # skip connections
#     skip_connections = []
#     for d in downsample_stack:
#         concat = d(concat)
#         skip_connections.append(concat)

#     skip_connections = reversed(skip_connections[:-1])

#     for up, skip in zip(upsample_stack, skip_connections):
#         concat = up(concat)
#         concat = tf.keras.layers.Concatenate()([concat, skip])

#     output = last(concat)

#     return tf.keras.Model(inputs=[sonar_input, camera_input], outputs=output)


# def PIX2PIX_DISCRIMINATOR_SONAR_CAMERA(
#     arch_type,
#     input_shape=[256, 512, 3],
#     output_shape=[256, 512, 3],
#     base_filters=256,
# ):
#     # define PatchGAN discriminator
#     sonar_input = tf.keras.layers.Input(input_shape)
#     camera_input = tf.keras.layers.Input(input_shape)
#     reference_input = tf.keras.layers.Input(input_shape)

#     sonar_input_conv = tf.keras.layers.Conv2D(
#         filters=256, kernel_size=3, strides=1, padding="same"
#     )(sonar_input)
#     camera_input_conv = tf.keras.layers.Conv2D(
#         filters=256, kernel_size=3, strides=1, padding="same"
#     )(camera_input)

#     # TODO: how to concatenate the inputs?
#     concat = tf.keras.layers.Concatenate()(
#         [sonar_input_conv, camera_input_conv, reference_input]
#     )

#     # TODO: how to go down till 70x140 what is equivalent?
#     d1 = CBRDownsample(
#         filters=base_filters, kernel_size=4, apply_batchnorm=False, leakyReLU=True
#     )(concat)
#     d2 = CBRDownsample(filters=base_filters * 2, kernel_size=4, leakyReLU=True)(d1)
#     d3 = CBRDownsample(filters=base_filters * 4, kernel_size=4, leakyReLU=True)(d2)
#     d4 = CBRDownsample(filters=base_filters * 8, kernel_size=4, leakyReLU=True)(d3)
#     d5 = CBRDownsample(filters=1, kernel_size=4, leakyReLU=True)(d4)

#     return tf.keras.Model(
#         inputs=[sonar_input, camera_input, reference_input], outputs=d5
#     )


def PIX2PIX_GENERATOR(
    arch_type,
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

    # TODO: change concat based on with-camera-early-fusion or with-camera-late-fusion
    if "with-camera-" in arch_type:
        sonar_input = tf.keras.layers.Input(input_shape)
    camera_input = tf.keras.layers.Input(input_shape)

    # TODO: change concat based on with-camera-early-fusion or with-camera-late-fusion -- check filters
    if "with-camera-" in arch_type:
        sonar_input_conv = tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=1, padding="same"
        )(sonar_input)
        camera_input_conv = tf.keras.layers.Conv2D(
            filters=16, kernel_size=3, strides=1, padding="same"
        )(camera_input)

    # TODO: change concat based on with-camera-early-fusion or with-camera-late-fusion
    if "with-camera-" in arch_type:
        concat = tf.keras.layers.Concatenate()([camera_input_conv, sonar_input_conv])
    else:
        concat = camera_input

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
        1, count + 1
    ):  # may be reduce this to keep trainable params in check
        num_filters = base_filters * (2 ** i) if base_filters * (2 ** i) <= 512 else 512
        downsample_stack.append(
            CBRDownsample(filters=num_filters, kernel_size=4, leakyReLU=True)
        )

    # create upsample stack of layers for coming back to output resolution
    upsample_stack = []
    for i in range(
        1, count - 3
    ):  # may be reduce this to keep trainable params in check
        upsample_stack.append(CBRUpsample(filters=512, kernel_size=4, leakyReLU=True))

    for num_filters in [256, 128, 64]:
        upsample_stack.append(
            CBRUpsample(filters=num_filters, kernel_size=4, leakyReLU=True)
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

    # TODO: change concat based on with-camera-early-fusion or with-camera-late-fusion
    if "with-camera-" in arch_type:
        return tf.keras.Model(inputs=[camera_input, sonar_input], outputs=output)
    else:
        return tf.keras.Model(inputs=camera_input, outputs=output)


def PIX2PIX_DISCRIMINATOR(
    arch_type,
    input_shape=[256, 512, 3],
    output_shape=[256, 512, 3],
    base_filters=256,
):
    # define PatchGAN discriminator
    # TODO: change concat based on with-camera-early-fusion or with-camera-late-fusion
    if "with-camera-" in arch_type:
        sonar_input = tf.keras.layers.Input(input_shape)
    camera_input = tf.keras.layers.Input(input_shape)
    reference_input = tf.keras.layers.Input(input_shape)

    initializer = tf.random_normal_initializer(0.0, 0.02)

    # TODO: change concat based on with-camera-early-fusion or with-camera-late-fusion
    if "with-camera-" in arch_type:
        sonar_input_conv = tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, strides=1, padding="same"
        )(sonar_input)
    
    camera_input_conv = tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, strides=1, padding="same"
        )(camera_input)
    reference_input_conv = tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, strides=1, padding="same"
        )(reference_input)


    # TODO: change concat based on with-camera-early-fusion or with-camera-late-fusion
    # concat = tf.keras.layers.Concatenate()()

    # TODO: how to go down till 70x140 what is equivalent?
    if "with-camera-" in arch_type:
        x = tf.keras.layers.concatenate(
            [camera_input_conv, sonar_input_conv, reference_input_conv]
        )
    else:
        x = tf.keras.layers.concatenate(
            [camera_input, reference_input]
        )  # (batch_size, 256, 256, channels*2)

    down1 = CBRDownsample(64, 2, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = CBRDownsample(128, 2, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = CBRDownsample(256, 2, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(
        512, 4, strides=1, kernel_initializer=initializer, use_bias=False
    )(
        zero_pad1
    )  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(
        zero_pad2
    )  # (batch_size, 30, 30, 1)

    if "with-camera-" in arch_type:
        return tf.keras.Model(
            inputs=[camera_input, sonar_input, reference_input], outputs=last
        )
    else:
        return tf.keras.Model(inputs=[camera_input, reference_input], outputs=last)
