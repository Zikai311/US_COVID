"""
Utility functions for epidemic modeling project.
"""

import numpy as np
from typing import Tuple, Optional


def ensure_array(x):
    """Convert input to numpy array if not already."""
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def normalize_state(S, I, R, N=None):
    """
    Normalize state variables to sum to N.

    Parameters
    ----------
    S : float or array
        Susceptible population
    I : float or array
        Infected population
    R : float or array
        Recovered population
    N : float or None
        Total population. If None, computed as S + I + R

    Returns
    -------
    S, I, R, N : tuple
        Normalized state variables and total population
    """
    S = ensure_array(S)
    I = ensure_array(I)
    R = ensure_array(R)

    if N is None:
        N = S + I + R

    return S, I, R, N


def compute_R0(beta, gamma):
    """
    Compute basic reproduction number R0.

    Parameters
    ----------
    beta : float
        Transmission rate
    gamma : float
        Recovery rate

    Returns
    -------
    R0 : float
        Basic reproduction number
    """
    return beta / gamma


def find_peak(t, I):
    """
    Find the peak infection time and value.

    Parameters
    ----------
    t : array
        Time array
    I : array
        Infected population array

    Returns
    -------
    peak_time : float
        Time at which infection peaks
    peak_value : float
        Peak infection value
    peak_index : int
        Index of peak in arrays
    """
    peak_index = np.argmax(I)
    peak_time = t[peak_index]
    peak_value = I[peak_index]

    return peak_time, peak_value, peak_index
