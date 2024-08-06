#!/usr/bin/env python3
import tensorflow as tf

def shear_image(image, intensity):
    """
    Randomly shears an image.

    Args:
    image (tf.Tensor): A 3D tf.Tensor containing the image to shear.
    intensity (int): The intensity with which the image should be sheared.

    Returns:
    tf.Tensor: The sheared image.
    """
    # Shear transformation matrix
    shear_matrix = [[1.0, intensity, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]]
    
    sheared_image = tf.keras.preprocessing.image.apply_affine_transform(image, shear=intensity)
    return sheared_image
