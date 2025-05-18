#!/usr/bin/env python3
"""Deep RNN forward propagation"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN

    Parameters:
    - rnn_cells: list of RNNCell instances
    - X: np.ndarray of shape (t, m, i) - input data
    - h_0: np.ndarray of shape (num_layers, m, h) - initial hidden state

    Returns:
    - H: np.ndarray of shape (t + 1, num_layers, m, h) - hidden states
    - Y: np.ndarray of shape (t, m, o) - outputs
    """
    t, m, _ = X.shape
    num_layers = len(rnn_cells)
    _, _, h = h_0.shape
    o = rnn_cells[-1].by.shape[1]  # output dimension from the last cell

    H = np.zeros((t + 1, num_layers, m, h))
    H[0] = h_0
    Y = []

    for time_step in range(t):
        x = X[time_step]
        for layer in range(num_layers):
            cell = rnn_cells[layer]
            h_prev = H[time_step, layer]
            h_next, y = cell.forward(h_prev, x)
            H[time_step + 1, layer] = h_next
            x = h_next  # input for the next layer is the hidden state
        Y.append(y)

    Y = np.array(Y)
    return H, Y
