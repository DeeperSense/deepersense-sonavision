import time
import tensorflow as tf


@tf.function
def train_step(
    input,
    target,
    step,
    generator,
    discriminator,
    generator_loss_fn,
    discriminator_loss_fn,
    generator_optimizer,
    discriminator_optimizer,
    lambda_l1,
    loss_object,
    summary_writer,
):
    with tf.GradientTape() as pix2pix_gen_tape, tf.GradientTape() as pix2pix_disc_tape:
        # get generator output
        gen_out = generator(input, training=True)
        # get discriminator outputs (from real, generated image)
        disc_real_out = discriminator([input, target], training=True)
        disc_generated_out = discriminator([input, gen_out], training=True)

        # calculate generator loss  disc_generated_output, gen_output, target, lambda_l1, loss_object
        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss_fn(
            disc_generated_out, gen_out, target, lambda_l1, loss_object
        )

        # calculate discriminator loss disc_real_output, disc_generated_output, loss_object
        disc_loss = discriminator_loss_fn(disc_real_out, disc_generated_out, loss_object)

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

    # write summary (also log images generated every 100th step?)
    with summary_writer.as_default():
        tf.summary.scalar("gen_total_loss", gen_total_loss, step=step // 1000)
        tf.summary.scalar("gen_gan_loss", gen_gan_loss, step=step // 1000)
        tf.summary.scalar("gen_l1_loss", gen_l1_loss, step=step // 1000)
        tf.summary.scalar("disc_loss", disc_loss, step=step // 1000)
        tf.summary.image("gen_out", gen_out, step=step // 1000)
        tf.summary.image("disc_real_out", disc_real_out, step=step // 1000)
        tf.summary.image("disc_generated_out", disc_generated_out, step=step // 1000)
        


def fit(train_dataset, steps, generator,discriminator,generator_loss_fn,discriminator_loss_fn,generator_optimizer,discriminator_optimizer,lambda_l1, loss_object,summary_writer, checkpoint, checkpoint_prefix):

    for step, (input, target) in train_dataset.repeat().take(steps).enumerate():
        tick = time.time()
        if step % 1000 == 0:
            tick = time.time()
            print(f"Step: {step//1000}k")
        train_step(input, target, step, generator,discriminator,generator_loss_fn,discriminator_loss_fn,generator_optimizer,discriminator_optimizer, lambda_l1, loss_object,summary_writer,)
        # training step
        if (step + 1) % 1 == 0:
            print(".", end="", flush=True)

        # Save (checkpoint) the model every 5k steps
        if (step + 1) % 1 == 0 and checkpoint and checkpoint_prefix:
            checkpoint.save(file_prefix=checkpoint_prefix)
        if  (step) % 1000 == 0 and step != 0:
            print(f"Time taken for 1000 steps: {time.time()-tick:.2f} sec\n")
        