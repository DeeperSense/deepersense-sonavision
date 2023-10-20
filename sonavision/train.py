import pathlib
import os
import pdb

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from options.pix2pix_tf import Pix2PixOptions
from utils.pix2pix_tf import load_image, generate_images, visualize_image
from models.pix2pix_tf import PIX2PIX_DISCRIMINATOR, PIX2PIX_GENERATOR
from models.double_encoder_unet_tf import DOUBLE_ENCODER_UNET

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

print("Number of training images:", len(train_dataset))

# pdb.set_trace()

train_dataset = train_dataset.map(
    lambda x: load_image(
        x,
        height=input_shape[0],
        width=input_shape[1],
        num_images_per_image=opt.num_images_per_image,
        image_format=opt.image_format,
        normalize=True,
    )
)

train_dataset = train_dataset.batch(opt.batch_size)

# define validation dataset
val_dataset = tf.data.Dataset.list_files(
    str(
        pathlib.Path(opt.dataset_dir)
        / pathlib.Path(opt.val_subdir)
        / pathlib.Path("*." + opt.image_format)
    )
)

print("Number of validation images:", len(val_dataset))

val_dataset = val_dataset.map(
    lambda x: load_image(
        x,
        height=input_shape[0],
        width=input_shape[1],
        num_images_per_image=opt.num_images_per_image,
        image_format=opt.image_format,
        normalize=True,
    )
)

# define define generator and discriminator optimizers
generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=opt.lrG, beta_1=opt.beta1, beta_2=opt.beta2
)
discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=opt.lrD, beta_1=opt.beta1, beta_2=opt.beta2
)
# define generator and discriminator models

if opt.arch_type == "with-camera-late-fusion":
    generator = DOUBLE_ENCODER_UNET(
        input_shape=input_shape,
        output_shape=output_shape,
        base_filters=opt.ngf,
    )
else:
    generator = PIX2PIX_GENERATOR(
        arch_type=opt.arch_type,
        input_shape=input_shape,
        output_shape=output_shape,
        base_filters=opt.ngf,
    )

discriminator = PIX2PIX_DISCRIMINATOR(
    arch_type=opt.arch_type,
    input_shape=input_shape,
    output_shape=output_shape,
    base_filters=opt.ndf,
)
# pdb.set_trace()
print("[INFO] Generator summary")
generator.summary()
print("[INFO] Discriminator summary")
discriminator.summary()
# define checkpoint manager
checkpoint_prefix = pathlib.Path(opt.checkpoints_dir) / "ckpt"
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
    opt.arch_type,
    train_dataset,
    val_dataset,
    opt.train_epoch,
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

generator.save(opt.model_save_dir)


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
        num_images_per_image=opt.num_images_per_image,
        normalize=True,
    )
)
test_dataset = test_dataset.batch(opt.batch_size)

for idx, (target, camera, sonar) in enumerate(test_dataset):
    generate_images(
        generator,
        camera,
        sonar,
        target,
        opt.results_dir / pathlib.Path(str(idx) + ".png"),
    )
