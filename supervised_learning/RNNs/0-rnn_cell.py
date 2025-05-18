#!/usr/bin/env python3
"""
Defines the RNNCell class for one time step of a simple RNN
"""

import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i (int): dimensionality of the data
            h (int): dimensionality of the hidden state
            o (int): dimensionality of the outputs
        """
        # Weights and biases for hidden state computation
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))

        # Weights and biases for output computation
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        Args:
            h_prev (ndarray): shape (m, h), previous hidden state
            x_t (ndarray): shape (m, i), input data at time t

        Returns:
            h_next: next hidden state, shape (m, h)
            y: output of the cell, shape (m, o)
        """
        # Concatenate previous hidden state and current input
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state using tanh activation
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        # Compute raw output
        y_linear = np.matmul(h_next, self.Wy) + self.by

        # Softmax activation for output
        y = self.softmax(y_linear)

        return h_next, y

    @staticmethod
    def softmax(x):
        """
        Compute softmax for each row of input x

        Args:
            x (ndarray): shape (m, o)

        Returns:
            Softmax probabilities
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
