from sonavision.building_blocks import CBR_upsample, CBR_downsample

def UNetSonarCamera(height=512, width=512, layers=(512,256,128,64)):
    """Modified U-net autoencoder that fuses camera and sonar images.

    Args:
        height (int): height of image defining shape of the input layer
        width (int): width of image defining shape of the input layer
        layers (tuple: int): size of layers or the encoder in order. The
        decoder is the inverse of the given order

    Returns:
        tf.keras.Model: generator model
    """
