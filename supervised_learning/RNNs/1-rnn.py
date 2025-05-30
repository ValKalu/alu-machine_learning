#!/usr/bin/env python3
""" Forward propagation for a simple RNN """

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Parameters:
    - rnn_cell: instance of RNNCell used for forward
      propagation
    - X: numpy.ndarray of shape (t, m, i),
      input data
    - h_0: numpy.ndarray of shape (m, h),
      initial hidden state

    Returns:
    - H: numpy.ndarray of shape (t+1, m, h),
      hidden states including initial state
    - Y: numpy.ndarray of shape (t, m, o),
      outputs for each time step
    """

    t, m, i = X.shape
    h = h_0.shape[1]

    H = np.zeros((t + 1, m, h))
    H[0] = h_0  # store initial hidden state

    Y = []

    h_t = h_0
    for time_step in range(t):
        x_t = X[time_step]
        h_t, y_t = rnn_cell.forward(h_t, x_t)
        H[time_step + 1] = h_t  # store hidden state for time step
        Y.append(y_t)

    Y = np.array(Y)

    return H, Y
