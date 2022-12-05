import tensorflow as tf

# squeezenet fire module
def Fire(x, s11, e11, e33, activation=tf.nn.relu, layer_id=None):
    """Defines a fire module for squeezenet. It is used iteratively when building squeezenet.
    more info: https://arxiv.org/pdf/1602.07360.pdf and https://paperswithcode.com/method/fire-module.

    Args:
        x (tf.Tensor): input tensor
        s11 (int): number of filters in squeeze layer
        e11 (int): number of filters in 1x1 expand layer
        e33 (int): number of filters in 3x3 expand layer
        activation (tf.nn.relu, optional): activation function. Defaults to tf.nn.relu.
        layer_id (int, optional): layer id. Defaults to None.

    Returns:
        tf.keras.Sequential: downsampling block
    """
    with tf.name_scope("fire" + str(layer_id)):
        s1 = tf.keras.layers.Conv2D(s11, (1, 1), padding="same", activation=activation)(
            x
        )
        e1 = tf.keras.layers.Conv2D(e11, (1, 1), padding="same", activation=activation)(
            s1
        )
        e3 = tf.keras.layers.Conv2D(e33, (3, 3), padding="same", activation=activation)(
            s1
        )
        concatenated = tf.keras.layers.concatenate([e1, e3], axis=-1)
        return tf.keras.layers.BatchNormalization()(concatenated)


