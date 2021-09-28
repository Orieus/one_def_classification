#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" This script is aimed at evaluating the behavior of tourney_prob, which
    computes the selection probabilities in a set of size N.

    The selection probability p for tuple (k, N, m) is the probability  of
    selecting the k-th best sample in a ranked set of N samples when taking the
    best sample in a random subset of m elements.

    - Tests if the function is correct (by testing if probabilities agree with
      frequencies of occurrence)
    - Evaluates the sample average estimates of E{1/p} and some related
      measures. This is an indicator of the behavior that can be expected from
      average statistics computed from active learning samplers based on
      tournaments (which is the the algorithm used by the WebLabeler
      application)

    Use: python test_psel.py

    Check configurable parameters.

    Author: JCS, Mar. 2016
"""

# External modules
import sys
import numpy as np
import matplotlib.pyplot as plt
import ipdb
from time import time


def tourney_prob(k, N, m, flag=1):

    """
    Compute the probability of the following event in the following experiment:
    - Experiment: given a set S_N of N distinct real numbers, take subset
                  S_m of m < N values at random without replacement.
    - Event: The highest value in S_m is the k-th highest value in S_N,
             given that the k-th higest value is in S_m

    Args:
        :k: Rank to be evaluated
        :N: Size of the original set
        :m: Size of the subset
        :flag: This is an internal parameter. It should not be used in the
               external calls to this function. The internal self-calls change
               the default value 1 to 0 to indicate that it is a recursive
               call so factor m cannot be applied

    Returns:
        :p: Probability of k-th highest value over N being the highest over m
    """

    if N < m:
        print("The second argument cannot be smaller than the third one.")
        sys.exit()

    if m > 1 and k > 0:
        if flag == 1:
            return m * float(N - k) / N * tourney_prob(k, N - 1, m - 1, flag=0)
        else:
            return float(N - k) / N * tourney_prob(k, N - 1, m - 1, flag=0)
    elif m == 1:
        return 1.0 / N
    else:
        return 0.0


def tourney_prob_block(N, m, kmin=1, kmax=None):

    """
    Compute the probability of the following event in the following experiment:
    - Experiment: given a set S_N of N distinct real numbers, take subset
                  S_m of m < N values at random without replacement.
    - Event: The highest value in S_m is the k-th highest value in S_N,
             given that the k-th higest value is in S_m
    k is explored from kmin to kmax

    Args:
        :N: Size of the original set
        :m: Size of the subset
        :kmin: Minimum value of k
        :kmax: Maximum value of k

    Returns:
        :p: Probability of k-th highest value over N being the highest over m
    """

    if kmax is None:
        kmax = N

    if N < m:
        print("The second argument cannot be smaller than the third one.")
        sys.exit()

    k = np.arange(kmin, kmax + 1)

    # This is valid for m=1 only. For m>1, it is just a factor.
    p = 1.0 / (N - m + 1) * np.ones(kmax - kmin + 1)

    if m > 1:
        # Use p as a factor and compute the rest of factors.
        for mu in range(m):
            Nu = N - mu
            if mu == m - 1:
                p = m * p
            else:
                p = (float(Nu) - k) / Nu * p

    return p


# Configurable parameters
ts = 780            # Tourney size
n_tot = 785898         # Dataset size
k_min = 1
k_max = 100

# ####################################################
# Testing tourney probabilities and importance weights

print("*****************")
print("* PSEL TEST *****")

print("This is a test to verify that the probabilities computed by " +
      "tourney_prob and tourney_prob_block are equal.")

# Testing p_sel
n = n_tot + ts - 1

print("\nComputation times:")

t0 = time()
p_sel = [tourney_prob(k, n, ts) for k in range(k_min, k_max + 1)]
print("---- 1by1 computation (based on the recursive function) in " +
      "{} seconds.".format(time() - t0))

t0 = time()
p_selc = [tourney_prob_block(n, ts, kmin=k, kmax=k)[0] for k in
          range(k_min, k_max + 1)]
print("---- 1by1 computation looping over the block method in " +
      "{} seconds.".format(time() - t0))

t0 = time()
p_selb = tourney_prob_block(n, ts, k_min, k_max)
print("---- Block computation in {} seconds.".format(time() - t0))

print("\nResult differences:")
# print("---- True selection probabilities: {}".format(p_sel))
# print("---- Block selection probabilities: {}".format(p_selb))

print("---- Maximum absolute difference: {}".format(np.max(p_selb-p_sel)))

x = list(range(1, len(p_sel) + 1))

plt.figure()
plt.plot(x, p_sel, label='True probabilities')
plt.plot(x, p_selb, label='Block probabilities')
plt.xlabel('$k$')
plt.ylabel('Selection Probability P($k$, {0}, {1})'.format(n, ts))
plt.legend()
plt.show(block=False)

plt.figure()
plt.plot(x, np.array(p_selb) - np.array(p_sel), label='Prob difference')
plt.xlabel('$k$')
plt.ylabel('Selection Probability P($k$, {0}, {1})'.format(n, ts))
plt.legend()
plt.show(block=False)

print("---- Minimum prob = {}".format(min([p for p in p_sel if p > 0])))

print("I am stopping just to keep figures open.")
ipdb.set_trace()

