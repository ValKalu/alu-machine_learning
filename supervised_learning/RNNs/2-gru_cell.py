#!/usr/bin/env python3
"""GRU Cell"""

import numpy as np


class GRUCell:
    """Represents a gated recurrent unit"""

    def __init__(self, i, h, o):
        """Class constructor
        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    @staticmethod
    def _sigmoid(x):
        """ Sigmoid activation function """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _softmax(x):
        """ Softmax activation function """
        exp_x = np.exp(
            x - np.max(x, axis=1, keepdims=True)
        )
        return exp_x / np.sum(
            exp_x, axis=1, keepdims=True
        )

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step
        h_prev: numpy.ndarray of shape (m, h) - previous hidden state
        x_t: numpy.ndarray of shape (m, i) - data input for the cell
        Returns: h_next, y
            h_next: next hidden state
            y: output of the cell
        """
        m = x_t.shape[0]

        # Concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Update gate
        z = self._sigmoid(
            np.matmul(concat, self.Wz) + self.bz
        )

        # Reset gate
        r = self._sigmoid(
            np.matmul(concat, self.Wr) + self.br
        )

        # Intermediate hidden state
        concat_r = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(
            np.matmul(concat_r, self.Wh) + self.bh
        )

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_tilde

        # Output
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = self._softmax(y_linear)

        return h_next, y
