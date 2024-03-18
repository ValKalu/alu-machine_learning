#!/usr/bin/env python3
"""Contains the posterior function"""

import numpy as np
from scipy import special

def posterior(x, n, p1, p2):
    """
    Calculates the posterior probability that the probability of developing severe side effects
    falls within a specific range given the data.

    Args:
        x (int): The number of patients that develop severe side effects.
        n (int): The total number of patients observed.
        p1 (float): The lower bound on the range.
        p2 (float): The upper bound on the range.

    Returns:
        float: The posterior probability that p is within the range [p1, p2] given x and n.

    Raises:
        ValueError: If n is not a positive integer, x is not a non-negative integer,
                    x is greater than n, p1 or p2 are not floats within [0, 1],
                    or p2 <= p1.
    """
    # Check if n is a positive integer
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check if x is a non-negative integer
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    # Check if x is greater than n
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check if p1 and p2 are floats within [0, 1]
    if not (0 <= p1 <= 1) or not (0 <= p2 <= 1):
        raise ValueError("p1 and p2 must be floats in the range [0, 1]")

    # Check if p2 > p1
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # Calculate the posterior using Beta distribution
    alpha = x + 1
    beta = n - x + 1
    posterior_prob = (special.betainc(alpha, beta, p2) - special.betainc(alpha, beta, p1)) / (special.beta(alpha, beta))
    return posterior_prob
