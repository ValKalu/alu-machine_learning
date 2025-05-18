#!/usr/bin/env python3
"""Bidirectional RNN Cell"""

import numpy as np


class BidirectionalCell:
    """Represents a bidirectional RNN cell"""

    def __init__(self, i, h, o):
        """
        Class constructor

        Args:
            i: dimensionality of the data
            h: dimensionality of the hidden states
            o: dimensionality of the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.randn(2 * h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction

        Args:
            h_prev: np.ndarray of shape (m, h) - previous hidden state
            x_t: np.ndarray of shape (m, i) - input data at time t

        Returns:
            h_next: np.ndarray of shape (m, h) - next hidden state
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)
        return h_next
