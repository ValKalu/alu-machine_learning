#!/usr/bin/env python3

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell

np.random.seed(1)
rnn = RNNCell(10, 15, 5)

x = np.random.randn(4, 10)
h = np.random.randn(4, 15)
h_next, y = rnn.forward(h, x)
print(h_next)
print(y)
