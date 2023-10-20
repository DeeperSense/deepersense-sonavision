import time
import tensorflow as tf
from utils.pix2pix_tf import visualize_image
import pdb
from tensorflow.keras.metrics import Mean

epoch_train_loss = Mean(name="epoch_train_loss")
epoch_val_loss = Mean(name="epoch_val_loss")


@tf.function
def step(
    input,
    target,
    generator,
    discriminator,
    generator_loss_fn,
    discriminator_loss_fn,
    generator_optimizer,
    discriminator_optimizer,
    lambda_l1,
    loss_object,
    allow_train,
):
    with tf.GradientTape() as pix2pix_gen_tape, tf.GradientTape() as pix2pix_disc_tape:
        # get generator output
        _input = (
            input
            if allow_train
            else [x[tf.newaxis, ...] for x in input]
            if isinstance(input, tuple)
            else input[tf.newaxis, ...]
        )
        _target = target if allow_train else target[tf.newaxis, ...]
        gen_out = generator(
            _input,
            training=allow_train,
        )
        # get discriminator outputs (from real, generated image)
        disc_real_out = discriminator(
            [_input, _target],
            training=allow_train,
        )
        disc_generated_out = discriminator(
            [_input, gen_out],
            training=allow_train,
        )
        # calculate generator loss  disc_generated_output, gen_output, target, lambda_l1, loss_object
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss_fn(
            disc_generated_out, gen_out, target, lambda_l1, loss_object
        )
        # calculate discriminator loss disc_real_output, disc_generated_output, loss_object
        disc_loss = discriminator_loss_fn(
            disc_real_out, disc_generated_out, loss_object
        )

    if allow_train:
        # get generator gradients
        gen_gradients = pix2pix_gen_tape.gradient(
            gen_total_loss, generator.trainable_variables
        )
        # get discriminator gradients
        disc_gradients = pix2pix_disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )
        # apply generator gradients
        generator_optimizer.apply_gradients(
            zip(gen_gradients, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(disc_gradients, discriminator.trainable_variables)
        )
        epoch_train_loss(gen_l1_loss)

    if not allow_train:
        epoch_val_loss(gen_l1_loss)

    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


def fit(
    arch_type,
    train_dataset,
    val_dataset,
    epochs,
    generator,
    discriminator,
    generator_loss_fn,
    discriminator_loss_fn,
    generator_optimizer,
    discriminator_optimizer,
    lambda_l1,
    loss_object,
    summary_writer,
    checkpoint,
    checkpoint_prefix,
):
    tick = time.time()

    for epoch in range(epochs):
        epoch_train_loss.reset_states()
        epoch_val_loss.reset_states()
        tick = time.time()
        print(f"Training epoch: {epoch}")
        for idx, sample in enumerate(train_dataset):
            input = sample[1:] if "with-camera-" in arch_type else sample[2]
            target = sample[0]
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = step(
                input,
                target,
                generator,
                discriminator,
                generator_loss_fn,
                discriminator_loss_fn,
                generator_optimizer,
                discriminator_optimizer,
                lambda_l1,
                loss_object,
                allow_train=True,
            )

            if (idx + 1) % 10 == 0:
                print(".", end="", flush=True)

        # do validation pass every epoch
        print("\n")
        print(f"Validation epoch: {epoch}")
        for idx, sample in enumerate(val_dataset):
            input = sample[1:] if "with-camera-" in arch_type else sample[2]
            target = sample[0]
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = step(
                input,
                target,
                generator,
                discriminator,
                generator_loss_fn,
                discriminator_loss_fn,
                generator_optimizer,
                discriminator_optimizer,
                lambda_l1,
                loss_object,
                allow_train=False,
            )
            if (idx + 1) % 10 == 0:
                print(".", end="", flush=True)

        time_taken = tick - time.time()
        # write summary of training loss
        with summary_writer.as_default():
            tf.summary.scalar("train/gen_total_loss", gen_total_loss, step=epoch)
            tf.summary.scalar("train/gen_gan_loss", gen_gan_loss, step=epoch)
            tf.summary.scalar("train/gen_l1_loss", gen_l1_loss, step=epoch)
            tf.summary.scalar("train/disc_loss", disc_loss, step=epoch)
            tf.summary.scalar(
                "train/mean_gen_l1_loss_epoch", epoch_train_loss.result(), step=epoch
            )
            tf.summary.scalar("val/mean_gen_l1_loss_epoch", epoch_val_loss.result(), step=epoch)

        template = "Epoch {}, Training Gen L1 Loss {:.4f}, Val Gen L1 Loss {:.4f}, Time Taken {:.2f} sec"
        print(
            template.format(
                epoch, epoch_train_loss.result(), epoch_val_loss.result(), time_taken
            )
        )

        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)
