import tensorflow as tf


def images_resize(images: list, height: int, width: int) -> list:
    """Resize images to the given size.

    Args:
        images (list): A list of images.
        height (int): The height of the resized images.
        width (int): The width of the resized images.

    Returns:
        A list of resized images.
    """
    resized_images = []
    for image in images:
        resized_images.append(
            tf.image.resize(
                image, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
            )
        )

    return resized_images


def random_crop(images: list, height: int, width: int) -> list:
    # stacked_image = tf.stack(images, axis=0)
    cropped_images = []
    for i in images:
        cropped_images.append(tf.image.random_crop(i, size=[2, height, width, 3]))

    return cropped_images


def normalize_inputs(x):
    normalized = []
    for img in x:
        normalized.append((img / 127.5) - 1)
    return normalized
