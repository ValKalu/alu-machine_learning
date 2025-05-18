#!/usr/bin/env python3
"""LSTM Cell"""

import numpy as np


class LSTMCell:
    """Represents an LSTM unit"""

    def __init__(self, i, h, o):
        """
        Class constructor
        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        # Forget gate weights and bias
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))

        # Update gate weights and bias
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))

        # Intermediate cell state weights and bias
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))

        # Output gate weights and bias
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))

        # Output weights and bias
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    @staticmethod
    def _sigmoid(x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _softmax(x):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        h_prev: numpy.ndarray of shape (m, h) - previous hidden state
        c_prev: numpy.ndarray of shape (m, h) - previous cell state
        x_t: numpy.ndarray of shape (m, i) - data input for the cell

        Returns: h_next, c_next, y
            h_next: next hidden state
            c_next: next cell state
            y: output of the cell
        """
        m = x_t.shape[0]

        # Concatenate h_prev and x_t for matrix multiplication
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f = self._sigmoid(np.matmul(concat, self.Wf) + self.bf)

        # Update gate
        u = self._sigmoid(np.matmul(concat, self.Wu) + self.bu)

        # Intermediate cell state (candidate)
        c_tilde = np.tanh(np.matmul(concat, self.Wc) + self.bc)

        # Next cell state
        c_next = f * c_prev + u * c_tilde

        # Output gate
        o = self._sigmoid(np.matmul(concat, self.Wo) + self.bo)

        # Next hidden state
        h_next = o * np.tanh(c_next)

        # Output
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = self._softmax(y_linear)

        return h_next, c_next, y
