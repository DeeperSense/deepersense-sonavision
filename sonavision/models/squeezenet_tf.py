import tensorflow as tf
from layers.squeezenet_tf import Fire


def SQUEEZENET(
    input_shape=[256, 512, 3],
    latent_space_dims=18,
    base_squeeze_filters=32,
    base_expand_filters=64,
):
    """Defines squeezenet model.
    more info: https://arxiv.org/pdf/1602.07360.pdf and https://paperswithcode.com/method/fire-module.

    Args:
        input_shape (list, optional): input shape. Defaults to [256, 512, 3].
        latent_space_dims (int, optional): latent space dimensions. Defaults to 18.
        base_squeeze_filters (int, optional): base number of filters in squeeze layer. Defaults to 32.
        base_expand_filters (int, optional): base number of filters in expand layer. Defaults to 64.

    Returns:
        tf.keras.Model: squeezenet model
    """
    inp = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(
        8, (7, 7), strides=(2, 2), padding="same", activation="relu"
    )(inp)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same")(x)

    x = Fire(
        x,
        s11=int(1.0 * base_squeeze_filters),
        e11=int(1.0 * base_expand_filters),
        e33=int(1.0 * base_expand_filters),
        layer_id=1,
    )
    x = Fire(
        x,
        s11=int(1.0 * base_squeeze_filters),
        e11=int(1.0 * base_expand_filters),
        e33=int(1.0 * base_expand_filters),
        layer_id=2,
    )
    x = Fire(
        x,
        s11=int(2.0 * base_squeeze_filters),
        e11=int(2.0 * base_expand_filters),
        e33=int(2.0 * base_expand_filters),
        layer_id=3,
    )

    x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same")(x)

    x = Fire(
        x,
        s11=int(2 * 1.0 * base_squeeze_filters),
        e11=int(2 * 1.0 * base_expand_filters),
        e33=int(2 * 1.0 * base_expand_filters),
        layer_id=4,
    )
    x = Fire(
        x,
        s11=int(2 * 1.5 * base_squeeze_filters),
        e11=int(2 * 1.5 * base_expand_filters),
        e33=int(2 * 1.5 * base_expand_filters),
        layer_id=5,
    )
    x = Fire(
        x,
        s11=int(2 * 1.5 * base_squeeze_filters),
        e11=int(2 * 1.5 * base_expand_filters),
        e33=int(2 * 1.5 * base_expand_filters),
        layer_id=6,
    )
    x = Fire(
        x,
        s11=int(2 * 2.0 * base_squeeze_filters),
        e11=int(2 * 2.0 * base_expand_filters),
        e33=int(2 * 2.0 * base_expand_filters),
        layer_id=7,
    )

    x = tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same")(x)

    x = Fire(
        x,
        s11=int(4 * 1.0 * base_squeeze_filters),
        e11=int(4 * 1.0 * base_expand_filters),
        e33=int(4 * 1.0 * base_expand_filters),
        layer_id=8,
    )

    x = tf.keras.layers.Conv2D(
        latent_space_dims, (5, 5), activation="relu", padding="same"
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    model = tf.keras.Model(inputs=inp, outputs=x)
    return model
