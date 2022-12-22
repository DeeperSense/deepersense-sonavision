import tensorflow as tf
from utils.common import images_resize, random_crop, normalize_inputs
from matplotlib import pyplot as plt



def decode_images(image_file, num_images: int, image_format: str = "png"):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)

    if image_format == "jpeg" or image_format == "jpg":
        image = tf.io.decode_jpeg(image)
    elif image_format == "png":
        image = tf.io.decode_png(image)

    # Split each image tensor into three tensors:
    # reference image, dark image, and sonar image
    w = tf.shape(image)[1]
    w = w // num_images

    images = []

    for i in range(num_images):
        images.append(tf.cast(image[:, i * w : (i + 1) * w, :], tf.float32))

    return images


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


def load_image(
    image_file,
    num_images_per_image: int = 3,
    crop: bool = False,
    normalize: bool = False,
    height: int = 256,
    width: int = 512,
):
    img = decode_images(image_file, num_images_per_image)
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


def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()