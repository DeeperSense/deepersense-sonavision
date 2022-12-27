import pathlib
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from options.pix2pix_tf import Pix2PixOptions
from utils.pix2pix_tf import load_image
from models.pix2pix_tf import (
    PIX2PIX_GENERATOR,
    PIX2PIX_DISCRIMINATOR,
)

opt = Pix2PixOptions().parse()
input_shape = opt.input_shape
output_shape = opt.output_shape

input_shape.append(3)
output_shape.append(3)

# configure GPU device and memory growth (GPU memory can complain)
if tf.config.list_physical_devices("GPU"):
    print("[INFO] Configuring physical GPU device")
    physical_devices = tf.config.experimental.list_physical_devices("GPU")
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

# define train dataset
train_dataset = tf.data.Dataset.list_files(
    str(
        pathlib.Path(opt.dataset_dir)
        / pathlib.Path(opt.train_subdir)
        / pathlib.Path("*." + opt.image_format)
    )
)

train_dataset = train_dataset.map(
    lambda x: load_image(
        x, height=input_shape[0], width=input_shape[1], num_images_per_image=2
    )
)

train_dataset = train_dataset.shuffle(400)
train_dataset = train_dataset.batch(opt.batch_size)


test_dataset = tf.data.Dataset.list_files(
    str(
        pathlib.Path(opt.dataset_dir)
        / pathlib.Path(opt.test_subdir)
        / pathlib.Path("*." + opt.image_format)
    )
)
test_dataset = test_dataset.map(
    lambda x: load_image(
        x,
        height=input_shape[0],
        width=input_shape[1],
        num_images_per_image=2,
        normalize=True,
    )
)
test_dataset = test_dataset.batch(opt.batch_size)

# define define generator and discriminator optimizers
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=opt.lrG, beta_1=opt.beta1, beta_2=opt.beta2
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=opt.lrD, beta_1=opt.beta1, beta_2=opt.beta2
)
# define generator and discriminator models

generator = PIX2PIX_GENERATOR(
    input_shape=input_shape,
    output_shape=output_shape,
    base_filters=opt.ngf,
)
discriminator = PIX2PIX_DISCRIMINATOR(
    input_shape=input_shape,
    output_shape=output_shape,
    base_filters=opt.ndf,
)

print("[INFO] Generator summary")
generator.summary()
print("[INFO] Discriminator summary")
discriminator.summary()
# define checkpoint manager
checkpoint_prefix = pathlib.Path(opt.checkpoint_dir) / "ckpt"
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

summary_writer = tf.summary.create_file_writer(
    str(pathlib.Path(opt.logs_dir) / "summary")
)

from train.pix2pix_tf import fit
from losses.pix2pix_tf import (
    discriminatorWithSonarCameraLoss,
    generatorWithSonarCameraLoss,
)

fit(
    train_dataset,
    40000,
    generator,
    discriminator,
    generatorWithSonarCameraLoss,
    discriminatorWithSonarCameraLoss,
    generator_optimizer,
    discriminator_optimizer,
    opt.lambda_l1,
    loss_object,
    summary_writer,
    checkpoint,
    checkpoint_prefix,
)

generator.save(opt.model_save_dir+"model")