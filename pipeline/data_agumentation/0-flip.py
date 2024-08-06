#!/usr/bin/env python3
import tensorflow as tf

def flip_image(image):
    """
    Flips an image horizontally.

    Args:
    image (tf.Tensor): A 3D tf.Tensor containing the image to flip.

    Returns:
    tf.Tensor: The flipped image.
    """
    return tf.image.flip_left_right(image)
