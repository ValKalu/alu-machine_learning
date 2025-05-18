#!/usr/bin/env python3
"""Forward propagation for a bidirectional RNN"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN

    Args:
        bi_cell: instance of BidirectionalCell
        X: np.ndarray of shape (t, m, i) containing the input data
        h_0: np.ndarray of shape (m, h) - initial hidden state (forward)
        h_t: np.ndarray of shape (m, h) - initial hidden state (backward)

    Returns:
        H: np.ndarray containing all of the concatenated hidden states
        Y: np.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    Hf = np.zeros((t, m, h))  # Forward hidden states
    Hb = np.zeros((t, m, h))  # Backward hidden states

    # Forward direction
    h_prev = h_0
    for time in range(t):
        h_next = bi_cell.forward(h_prev, X[time])
        Hf[time] = h_next
        h_prev = h_next

    # Backward direction
    h_next = h_t
    for time in reversed(range(t)):
        h_prev = bi_cell.backward(h_next, X[time])
        Hb[time] = h_prev
        h_next = h_prev

    # Concatenate forward and backward hidden states
    H = np.concatenate((Hf, Hb), axis=2)  # Shape: (t, m, 2*h)

    # Compute output using bi_cell
    Y = bi_cell.output(H)  # Shape: (t, m, o)

    return H, Y
