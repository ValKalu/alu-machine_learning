#!/usr/bin/env python3
"""Bidirectional RNN Cell with forward, backward, and output"""

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

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction

        Args:
            h_next: np.ndarray of shape (m, h) - next hidden state
            x_t: np.ndarray of shape (m, i) - input data at time t

        Returns:
            h_prev: np.ndarray of shape (m, h) - previous hidden state
        """
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(concat, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """
        Calculates all outputs for the RNN

        Args:
            H: np.ndarray of shape (t, m, 2 * h) - concatenated hidden states

        Returns:
            Y: np.ndarray of shape (t, m, o) - outputs
        """
        t, m, _ = H.shape
        Y = []

        for time_step in range(t):
            h_t = H[time_step]
            y_t = np.matmul(h_t, self.Wy) + self.by
            y_t = np.exp(y_t)
            y_t /= np.sum(y_t, axis=1, keepdims=True)
            Y.append(y_t)

        return np.array(Y)
