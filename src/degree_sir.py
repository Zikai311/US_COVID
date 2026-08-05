"""
Degree-based SIR epidemic model implementation.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Tuple, Optional


def compute_degree_distribution(graph):
    """
    Compute degree distribution from a network.

    Parameters
    ----------
    graph : networkx.Graph
        Input network

    Returns
    -------
    degrees : array
        Unique degree values
    P_k : array
        Probability of each degree P(k)
    """
    degrees = np.array([d for n, d in graph.degree()])
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    P_k = counts / len(degrees)

    return unique_degrees, P_k


def compute_theta(I_k, P_k, k_values, mean_k):
    """
    Compute theta: probability that a randomly chosen edge leads to an infected node.

    θ = (1/<k>) * Σ_k k * P(k) * I_k

    Parameters
    ----------
    I_k : array
        Infected fraction for each degree class
    P_k : array
        Degree distribution
    k_values : array
        Degree values
    mean_k : float
        Mean degree <k>

    Returns
    -------
    theta : float
        Edge-to-infected probability
    """
    if mean_k == 0:
        return 0.0
    return np.sum(k_values * P_k * I_k) / mean_k


def degree_rhs(t, state, beta, gamma, k_values, P_k, mean_k, N_classes):
    """
    Right-hand side of degree-based SIR equations.

    dS_k/dt = -beta * k * S_k * θ
    dI_k/dt = beta * k * S_k * θ - gamma * I_k
    dR_k/dt = gamma * I_k

    Parameters
    ----------
    t : float
        Time
    state : array
        Flattened [S_0, S_1, ..., I_0, I_1, ..., R_0, R_1, ...] vector
    beta : float
        Transmission rate per contact
    gamma : float
        Recovery rate
    k_values : array
        Degree values
    P_k : array
        Degree distribution
    mean_k : float
        Mean degree
    N_classes : int
        Number of degree classes

    Returns
    -------
    derivatives : array
        Time derivatives of state
    """
    # Unpack state
    S_k = state[0:N_classes]
    I_k = state[N_classes:2*N_classes]
    R_k = state[2*N_classes:3*N_classes]

    # Compute theta
    theta = compute_theta(I_k, P_k, k_values, mean_k)

    # Compute derivatives
    dS_k = -beta * k_values * S_k * theta
    dI_k = beta * k_values * S_k * theta - gamma * I_k
    dR_k = gamma * I_k

    return np.concatenate([dS_k, dI_k, dR_k])


def simulate_degree_model(beta, gamma, k_values, P_k, I0_k=None,
                          t_span=(0, 160), t_eval=None):
    """
    Simulate degree-based SIR model.

    Parameters
    ----------
    beta : float
        Transmission rate per contact
    gamma : float
        Recovery rate
    k_values : array
        Degree values
    P_k : array
        Degree distribution P(k)
    I0_k : array, optional
        Initial infected fraction for each degree class
        If None, small uniform infection across all classes
    t_span : tuple
        (start_time, end_time)
    t_eval : array, optional
        Times at which to evaluate solution

    Returns
    -------
    solution : OdeResult
        Solution object
    mean_k : float
        Mean degree
    """
    N_classes = len(k_values)
    mean_k = np.sum(k_values * P_k)

    # Initial conditions
    if I0_k is None:
        I0_k = np.ones(N_classes) * 0.01  # 1% infected in each class

    S0_k = 1.0 - I0_k  # Assume no initial recovered
    R0_k = np.zeros(N_classes)

    initial_state = np.concatenate([S0_k, I0_k, R0_k])

    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)

    solution = solve_ivp(
        degree_rhs,
        t_span,
        initial_state,
        args=(beta, gamma, k_values, P_k, mean_k, N_classes),
        t_eval=t_eval,
        method='RK45'
    )

    return solution, mean_k


def extract_degree_classes(solution, N_classes):
    """
    Extract S_k, I_k, R_k from solution.

    Parameters
    ----------
    solution : OdeResult
        Solution from simulate_degree_model
    N_classes : int
        Number of degree classes

    Returns
    -------
    S_k : array (N_classes, n_timepoints)
    I_k : array (N_classes, n_timepoints)
    R_k : array (N_classes, n_timepoints)
    """
    S_k = solution.y[0:N_classes, :]
    I_k = solution.y[N_classes:2*N_classes, :]
    R_k = solution.y[2*N_classes:3*N_classes, :]

    return S_k, I_k, R_k


def aggregate_by_degree(solution, P_k, N_classes):
    """
    Aggregate degree classes to get total S, I, R.

    Parameters
    ----------
    solution : OdeResult
        Solution from simulate_degree_model
    P_k : array
        Degree distribution
    N_classes : int
        Number of degree classes

    Returns
    -------
    S_total : array
        Total susceptible over time
    I_total : array
        Total infected over time
    R_total : array
        Total recovered over time
    """
    S_k, I_k, R_k = extract_degree_classes(solution, N_classes)

    # Weight by degree distribution
    S_total = np.sum(P_k[:, np.newaxis] * S_k, axis=0)
    I_total = np.sum(P_k[:, np.newaxis] * I_k, axis=0)
    R_total = np.sum(P_k[:, np.newaxis] * R_k, axis=0)

    return S_total, I_total, R_total


def plot_degree_distribution(k_values, P_k, title="Degree Distribution",
                             distribution_type="", ax=None):
    """
    Plot degree distribution.

    Parameters
    ----------
    k_values : array
        Degree values
    P_k : array
        Probability of each degree
    title : str
        Plot title
    distribution_type : str
        Type description (e.g., "Poisson", "Power Law")
    ax : matplotlib axes, optional
        Axes to plot on

    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(k_values, P_k, color='#4A90E2', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Degree k', fontsize=12)
    ax.set_ylabel('P(k)', fontsize=12)

    title_text = title
    if distribution_type:
        title_text += f' ({distribution_type})'

    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    mean_k = np.sum(k_values * P_k)
    ax.axvline(mean_k, color='red', linestyle='--', linewidth=2,
               label=f'⟨k⟩ = {mean_k:.2f}')
    ax.legend(fontsize=11)

    return ax


def plot_infection_by_degree(solution, k_values, P_k, N_classes,
                             time_points=None, title="Infection by Degree Class",
                             ax=None):
    """
    Plot infected fraction over time for different degree classes.

    Parameters
    ----------
    solution : OdeResult
        Solution from simulate_degree_model
    k_values : array
        Degree values
    P_k : array
        Degree distribution
    N_classes : int
        Number of degree classes
    time_points : list, optional
        Specific time indices to plot (if None, plot all)
    title : str
        Plot title
    ax : matplotlib axes, optional
        Axes to plot on

    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    S_k, I_k, R_k = extract_degree_classes(solution, N_classes)
    t = solution.t

    # Plot a subset of time points or specific degree classes
    if time_points is None:
        # Plot a few representative degree classes over time
        degree_indices = [0, N_classes//4, N_classes//2, 3*N_classes//4, N_classes-1]
        degree_indices = [i for i in degree_indices if i < N_classes]

        for idx in degree_indices:
            ax.plot(t, I_k[idx, :], label=f'k={k_values[idx]:.0f}', linewidth=2)

        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Infected Fraction', fontsize=12)
        ax.legend(fontsize=11, title='Degree')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    return ax
