import tensorflow as tf
from utils.pix2pix_tf import load_image
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
import pdb
import time
import argparse
import glob


def generate_images(
    model, test_input_camera, test_input_sonar, target, timestamp, how_to_save, filename
):
    prediction = model([test_input_camera, test_input_sonar], training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input_sonar[0], test_input_camera[0], target[0], prediction[0]]
    title = [
        "sonar image",
        "night image",
        "camera (original) image",
        "reconstructed image",
    ]
    if how_to_save == "detailed" or how_to_save == "both":
        path = timestamp + "/results/" + dark_mode + "/detailed/"
        if not os.path.isdir(path):
            os.makedirs(path)
        for i in range(4):
            plt.subplot(1, 4, i + 1)
            plt.title(title[i])
            # pixel values to 0,1 to plot
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis("off")
            plt.savefig(path + filename, bbox_inches="tight")

    if how_to_save == "concate" or how_to_save == "both":
        path = timestamp + "/results/" + dark_mode + "/concate/"
        if not os.path.isdir(path):
            os.makedirs(path)
        pred = prediction[0].numpy()
        camera_image = target.numpy()[0]
        stacked_image = np.hstack((pred, camera_image))
        # pdb.set_trace()
        matplotlib.image.imsave(path + filename, stacked_image * 0.5 + 0.5)


parser = argparse.ArgumentParser(description="Pix2Pix for Nightvision")
parser.add_argument("--f", type=int, default=0)
args = parser.parse_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

sonar_data = 1
type = "rect-sonar-starnberg-rotate-every-3-rgb"
dark_mode = "mixed-darkness"
how_to_save = "both"
grayscale = 0
timestamp = "2023-09-28-15-40-32"
# experiment_desc = "only-dark-75"
model_dir = "{}/models".format(timestamp)
batch_size = 1


if type == "nightvision":
    NETWORK_IN_WIDTH = 256
    NETWORK_IN_HEIGHT = 512
    PATH_data = "dataset/nightvision/"
elif (
    type == "dfki-divers-gemini"
    or type == "dfki-divers-oculus-new"
    or type == "dfki-divers-hemmoor"
    or type == "rect-sonar-hemmoor"
    or type == "rect-sonar-starnberg-rotate-every-3-rgb"
):
    NETWORK_IN_WIDTH = 1024
    NETWORK_IN_HEIGHT = 512
    PATH_data = "dataset/{}/".format(type)
elif (
    type == "dfki-divers-oculus"
    or type == "dfki-divers-hemmoor-gimp-highfreq"
    or type == "dfki-divers-oculus-sorted-every-3"
):
    NETWORK_IN_HEIGHT = NETWORK_IN_WIDTH = 512
    PATH_data = "dataset/{}/".format(type)
elif (
    type == "dfki-divers-hemmoor-halved"
    or type == "dfki-divers-hemmoor-halved-gimp"
    or type == "dfki-divers-hemmoor-halved-cropped"
):
    NETWORK_IN_HEIGHT = 256
    NETWORK_IN_WIDTH = 512
    PATH_data = "dataset/{}/".format(type)

print("[INFO] Defining test dataset")

filenames = sorted(glob.glob(PATH_data + dark_mode + "/test/*"))
# filenames = sorted(glob.glob(PATH_data +  "/dark100/test/*"))

test_dataset = tf.data.Dataset.from_tensor_slices(filenames)
test_dataset = test_dataset.map(
    lambda x: load_image(
        x,
        height=NETWORK_IN_HEIGHT,
        width=NETWORK_IN_WIDTH,
        num_images_per_image=3,
        normalize=True,
    )
)
test_dataset = test_dataset.batch(batch_size, drop_remainder=True)


print("[INFO] Importing model")
generator = tf.keras.models.load_model(model_dir)

print("[INFO] Translating sonar to camera image with pre-trained model")
last_print_t = 0
for idx, ((camera, night, sonar), filename) in enumerate(
    zip(test_dataset, filenames[0::batch_size])
):
    name = filename.split("/")[-1]
    if (time.time() - last_print_t) >= 60 * 5:
        last_print_t = time.time()
        print("saving image num: ", idx, " filename: ", name)
    sonar_data = 1
    if sonar_data == 1:
        generate_images(generator, night, sonar, camera, timestamp, how_to_save, name)
