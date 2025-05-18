#!/usr/bin/env python3
""" GRU Cell """

import numpy as np


class GRUCell:
    """ Gated Recurrent Unit cell """

    def __init__(self, i, h, o):
        """
        Initialize the GRU cell

        Args:
        - i (int): dimensionality of the data input
        - h (int): dimensionality of the hidden state
        - o (int): dimensionality of the output

        Public instance attributes:
        - Wz, Wr, Wh, Wy: weights matrices
        - bz, br, bh, by: bias vectors
        """
        # Update gate weights and bias
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))

        # Reset gate weights and bias
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

        # Intermediate hidden state weights and bias
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))

        # Output weights and bias
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step

        Args:
        - h_prev (np.ndarray): previous hidden state of shape (m, h)
        - x_t (np.ndarray): input data at time t of shape (m, i)

        Returns:
        - h_next (np.ndarray): next hidden state of shape (m, h)
        - y (np.ndarray): output of the cell of shape (m, o)
        """
        m, _ = x_t.shape

        # Concatenate h_prev and x_t on last axis for gate calculations
        concat = np.concatenate((h_prev, x_t), axis=1)  # shape (m, h + i)

        # Update gate
        z = self._sigmoid(np.dot(concat, self.Wz) + self.bz)  # (m, h)

        # Reset gate
        r = self._sigmoid(np.dot(concat, self.Wr) + self.br)  # (m, h)

        # Calculate candidate hidden state
        # For candidate, concatenate (r * h_prev) and x_t
        r_h_prev = r * h_prev
        concat_candidate = np.concatenate((r_h_prev, x_t), axis=1)  # (m, h + i)
        h_tilde = np.tanh(np.dot(concat_candidate, self.Wh) + self.bh)  # (m, h)

        # Next hidden state
        h_next = (1 - z) * h_prev + z * h_tilde  # (m, h)

        # Output (using softmax)
        y_linear = np.dot(h_next, self.Wy) + self.by  # (m, o)
        y = self._softmax(y_linear)  # (m, o)

        return h_next, y

    @staticmethod
    def _sigmoid(x):
        """ Sigmoid activation function """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _softmax(x):
        """ Softmax activation function """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
