"""
Network-based SIR epidemic model implementation.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def generate_complete_graph(n):
    """Generate a complete graph where every node is connected to every other."""
    return nx.complete_graph(n)


def generate_random_graph(n, p):
    """
    Generate an Erdős-Rényi random graph.

    Parameters
    ----------
    n : int
        Number of nodes
    p : float
        Probability of edge creation
    """
    return nx.erdos_renyi_graph(n, p)


def generate_small_world(n, k, p):
    """
    Generate a Watts-Strogatz small-world graph.

    Parameters
    ----------
    n : int
        Number of nodes
    k : int
        Each node is connected to k nearest neighbors
    p : float
        Rewiring probability
    """
    return nx.watts_strogatz_graph(n, k, p)


def generate_scale_free(n, m):
    """
    Generate a Barabási-Albert scale-free graph.

    Parameters
    ----------
    n : int
        Number of nodes
    m : int
        Number of edges to attach from a new node to existing nodes
    """
    return nx.barabasi_albert_graph(n, m)


class NetworkSIR:
    """
    SIR epidemic model on a network.
    """

    def __init__(self, graph, beta, gamma, initial_infected=None):
        """
        Initialize network SIR model.

        Parameters
        ----------
        graph : networkx.Graph
            Contact network
        beta : float
            Transmission probability per contact
        gamma : float
            Recovery probability per time step
        initial_infected : list or int, optional
            List of initially infected nodes, or number to randomly infect
        """
        self.graph = graph.copy()
        self.beta = beta
        self.gamma = gamma
        self.n_nodes = graph.number_of_nodes()

        # Initialize all nodes as susceptible
        for node in self.graph.nodes():
            self.graph.nodes[node]['state'] = 'S'

        # Set initial infections
        if initial_infected is None:
            initial_infected = 1

        if isinstance(initial_infected, int):
            initial_infected = np.random.choice(
                list(self.graph.nodes()),
                size=initial_infected,
                replace=False
            )

        for node in initial_infected:
            self.graph.nodes[node]['state'] = 'I'

        # Initialize history
        self.history = {
            'S': [self._count_state('S')],
            'I': [self._count_state('I')],
            'R': [self._count_state('R')]
        }

    def _count_state(self, state):
        """Count nodes in given state."""
        return sum(1 for node in self.graph.nodes()
                   if self.graph.nodes[node]['state'] == state)

    def step(self):
        """
        Perform one time step of the epidemic.

        Returns
        -------
        counts : dict
            Number of S, I, R individuals after this step
        """
        # Lists to track state changes (to avoid modifying during iteration)
        newly_infected = []
        newly_recovered = []

        # Process infections
        for node in self.graph.nodes():
            if self.graph.nodes[node]['state'] == 'I':
                # Try to infect neighbors
                for neighbor in self.graph.neighbors(node):
                    if self.graph.nodes[neighbor]['state'] == 'S':
                        if np.random.random() < self.beta:
                            newly_infected.append(neighbor)

                # Try to recover
                if np.random.random() < self.gamma:
                    newly_recovered.append(node)

        # Apply state changes
        for node in newly_infected:
            self.graph.nodes[node]['state'] = 'I'

        for node in newly_recovered:
            self.graph.nodes[node]['state'] = 'R'

        # Record counts
        counts = {
            'S': self._count_state('S'),
            'I': self._count_state('I'),
            'R': self._count_state('R')
        }

        self.history['S'].append(counts['S'])
        self.history['I'].append(counts['I'])
        self.history['R'].append(counts['R'])

        return counts

    def simulate(self, max_steps=200):
        """
        Run the epidemic until no more infections or max steps reached.

        Parameters
        ----------
        max_steps : int
            Maximum number of time steps

        Returns
        -------
        history : dict
            Time series of S, I, R counts
        """
        for _ in range(max_steps):
            counts = self.step()
            if counts['I'] == 0:
                break

        return self.history

    def get_state_nodes(self, state):
        """
        Get list of nodes in a given state.

        Parameters
        ----------
        state : str
            'S', 'I', or 'R'

        Returns
        -------
        nodes : list
            List of node IDs in the specified state
        """
        return [node for node in self.graph.nodes()
                if self.graph.nodes[node]['state'] == state]

    def plot_network(self, ax=None, pos=None, node_size=100):
        """
        Plot the network with nodes colored by state.

        Parameters
        ----------
        ax : matplotlib axes, optional
            Axes to plot on
        pos : dict, optional
            Node positions (if None, computed with spring layout)
        node_size : int
            Size of nodes

        Returns
        -------
        ax : matplotlib axes
        pos : dict
            Node positions (can be reused for consistent layouts)
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))

        if pos is None:
            pos = nx.spring_layout(self.graph, seed=42)

        # Get nodes by state
        s_nodes = self.get_state_nodes('S')
        i_nodes = self.get_state_nodes('I')
        r_nodes = self.get_state_nodes('R')

        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, alpha=0.2, ax=ax)

        # Draw nodes by state
        nx.draw_networkx_nodes(self.graph, pos, nodelist=s_nodes,
                               node_color='#4A90E2', node_size=node_size,
                               label='Susceptible', ax=ax)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=i_nodes,
                               node_color='#E94B3C', node_size=node_size,
                               label='Infected', ax=ax)
        nx.draw_networkx_nodes(self.graph, pos, nodelist=r_nodes,
                               node_color='#50C878', node_size=node_size,
                               label='Recovered', ax=ax)

        ax.axis('off')
        ax.legend(fontsize=11)

        return ax, pos


def plot_network_timeseries(history, title="Network SIR", ax=None):
    """
    Plot S, I, R time series from network simulation.

    Parameters
    ----------
    history : dict
        Dictionary with 'S', 'I', 'R' keys containing time series
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

    t = np.arange(len(history['S']))

    ax.plot(t, history['S'], label='Susceptible', linewidth=2, color='#4A90E2')
    ax.plot(t, history['I'], label='Infected', linewidth=2, color='#E94B3C')
    ax.plot(t, history['R'], label='Recovered', linewidth=2, color='#50C878')

    ax.set_xlabel('Time (steps)', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return ax
