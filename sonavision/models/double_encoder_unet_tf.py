import tensorflow as tf
from layers.unet_tf import CBRDownsample, CBRUpsample
import pdb
def DOUBLE_ENCODER_UNET(
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
    sonar = tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding="same"
    )(sonar_input)
    camera = tf.keras.layers.Conv2D(
        filters=16, kernel_size=3, strides=1, padding="same"
    )(camera_input)

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
    downsample_stack_camera = []
    for i in range(
        1, count + 1
    ):  # may be reduce this to keep trainable params in check
        num_filters = base_filters * (2 ** i) if base_filters * (2 ** i) <= 512 else 512
        downsample_stack_camera.append(
            CBRDownsample(filters=num_filters, kernel_size=4, leakyReLU=True)
        )

    downsample_stack_sonar = []
    for i in range(
        1, count + 1
    ):  # may be reduce this to keep trainable params in check
        num_filters = base_filters * (2 ** i) if base_filters * (2 ** i) <= 512 else 512
        downsample_stack_sonar.append(
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

    skip_connections = []
    
    for dc, ds in zip(downsample_stack_camera, downsample_stack_sonar):
        camera = dc(camera)
        sonar = ds(sonar)
        skip_connections.append(tf.keras.layers.Concatenate()([camera, sonar]))

    concat = tf.keras.layers.Concatenate()([camera, sonar])

    skip_connections = reversed(skip_connections[:-1])

    for up, skip in zip(upsample_stack, skip_connections):
        concat = up(concat)
        concat = tf.keras.layers.Concatenate()([concat, skip])

    output = last(concat)


    return tf.keras.Model(inputs=[camera_input, sonar_input], outputs=output)
