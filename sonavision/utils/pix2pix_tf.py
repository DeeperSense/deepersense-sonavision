import tensorflow as tf
from utils.common import images_resize, random_crop, normalize_inputs


def load_triplet(image_file, image_format: str = "png"):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)

    if image_format == "jpeg" or image_format == "jpg":
        image = tf.io.decode_jpeg(image)
    elif image_format == "png":
        image = tf.io.decode_png(image)

    # Split each image tensor into three tensors:
    # reference image, dark image, and sonar image
    w = tf.shape(image)[1]
    w = w // 3
    reference_image = image[:, :w, :]
    night_image = image[:, w : 2 * w, :]
    sonar_image = image[:, 2 * w :, :]

    # Convert both images to float32 tensors
    reference_image = tf.cast(reference_image, tf.float32)
    night_image = tf.cast(night_image, tf.float32)
    sonar_image = tf.cast(sonar_image, tf.float32)

    return reference_image, night_image, sonar_image


def random_jitter(images: list, height: int = 256, width: int = 256):
    # Resizing to height x width
    imgs = images_resize(images, height, width)

    # Random cropping back to 256x256
    imgs = random_crop(imgs, height, width)

    out = []
    if tf.random.uniform(()) > 0.5:
        for img in imgs:
            out.append(tf.image.flip_left_right(img))
    else:
        out = imgs

    return out


def load_image_train(
    image_file,
    crop: bool = False,
    normalize: bool = False,
    height: int = 256,
    width: int = 512,
):
    img = load_triplet(image_file)
    if crop:
        img = random_crop(img, height, width)
    if normalize:
        img = normalize_inputs(img)

    return img


def generate_images(
    model,
    reference_image,
    night_image,
    sonar_image,
    filename,
    timestamp,
    dark_mode,
    how_to_save="detailed",
):
    # prediction = model(test_input, training=True)
    # plt.figure(figsize=(15, 15))

    # display_list = [test_input[0], tar[0], prediction[0]]
    # title = ["Input Image", "Ground Truth", "Predicted Image"]

    # for i in range(3):
    #     plt.subplot(1, 3, i + 1)
    #     plt.title(title[i])
    #     # Getting the pixel values between [0, 1] to plot it.
    #     plt.imshow(display_list[i] * 0.5 + 0.5)
    #     plt.axis("off")
    # plt.savefig(save_path)
    # plt.show()
    pass
