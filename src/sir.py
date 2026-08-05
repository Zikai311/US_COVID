"""
SIR (Susceptible-Infected-Recovered) epidemic model implementation.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import Tuple, Optional

try:
    from .utils import compute_R0, find_peak
except ImportError:
    from utils import compute_R0, find_peak


def sir_rhs(t, state, beta, gamma):
    """
    Right-hand side of SIR differential equations.

    dS/dt = -beta * S * I
    dI/dt = beta * S * I - gamma * I
    dR/dt = gamma * I

    Parameters
    ----------
    t : float
        Time (not used, required by solve_ivp)
    state : array-like
        [S, I, R] state vector
    beta : float
        Transmission rate
    gamma : float
        Recovery rate

    Returns
    -------
    derivatives : list
        [dS/dt, dI/dt, dR/dt]
    """
    S, I, R = state
    N = S + I + R

    dS = -beta * S * I / N
    dI = beta * S * I / N - gamma * I
    dR = gamma * I

    return [dS, dI, dR]


def simulate_sir(beta, gamma, S0, I0, R0, t_span=(0, 160), t_eval=None):
    """
    Simulate the SIR model.

    Parameters
    ----------
    beta : float
        Transmission rate
    gamma : float
        Recovery rate
    S0 : float
        Initial susceptible population
    I0 : float
        Initial infected population
    R0 : float
        Initial recovered population
    t_span : tuple, optional
        (start_time, end_time) for simulation
    t_eval : array-like, optional
        Specific times at which to return solution

    Returns
    -------
    solution : OdeResult
        Solution object from solve_ivp
        Access time with solution.t
        Access [S, I, R] with solution.y
    """
    initial_state = [S0, I0, R0]

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)

    solution = solve_ivp(
        sir_rhs,
        t_span,
        initial_state,
        args=(beta, gamma),
        t_eval=t_eval,
        method='RK45',
        dense_output=True
    )

    return solution


def plot_sir(solution, beta=None, gamma=None, title="SIR Model",
             show_peak=False, ax=None):
    """
    Plot SIR trajectories.

    Parameters
    ----------
    solution : OdeResult
        Solution from simulate_sir
    beta : float, optional
        Transmission rate (for display)
    gamma : float, optional
        Recovery rate (for display)
    title : str
        Plot title
    show_peak : bool
        Whether to annotate infection peak
    ax : matplotlib axes, optional
        Axes to plot on

    Returns
    -------
    fig : matplotlib figure (if ax is None)
    ax : matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = None

    t = solution.t
    S, I, R = solution.y

    ax.plot(t, S, label='Susceptible', linewidth=2, color='#4A90E2')
    ax.plot(t, I, label='Infected', linewidth=2, color='#E94B3C')
    ax.plot(t, R, label='Recovered', linewidth=2, color='#50C878')

    if show_peak:
        peak_time, peak_value, _ = find_peak(t, I)
        ax.plot(peak_time, peak_value, 'o', color='#E94B3C',
                markersize=8, zorder=5)
        ax.annotate(f'Peak: {peak_value:.1f}\nat t={peak_time:.1f}',
                   xy=(peak_time, peak_value),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.5',
                            facecolor='white', alpha=0.8),
                   arrowprops=dict(arrowstyle='->',
                                  connectionstyle='arc3,rad=0'))

    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)

    title_text = title
    if beta is not None and gamma is not None:
        R0_val = compute_R0(beta, gamma)
        title_text += f' (β={beta:.2f}, γ={gamma:.2f}, R₀={R0_val:.2f})'

    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t[0], t[-1])

    if fig is not None:
        plt.tight_layout()
        return fig, ax
    return ax
