#!/usr/bin/env python3
"""Neural Style Transfer Class"""

import numpy as np
import tensorflow as tf

class NST:
    """Performs Neural Style Transfer"""
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initializes NST with the given style and content images"""
        if not isinstance(style_image, np.ndarray) or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Enable eager execution for TensorFlow
        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """Rescales an image such that its pixel values are between 0 and 1 
        and its largest side is 512 pixels"""
        if not isinstance(image, np.ndarray) or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        # Get image dimensions and scale proportionately
        h, w, _ = image.shape
        if h > w:
            new_h = 512
            new_w = int(w * (512 / h))
        else:
            new_w = 512
            new_h = int(h * (512 / w))

        # Resize image using bicubic interpolation and normalize pixel values
        image = tf.image.resize_bicubic([image], [new_h, new_w])
        image = image / 255.0

        return image
