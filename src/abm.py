"""
Agent-Based Model (ABM) for epidemic simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict


class Agent:
    """
    Individual agent in the epidemic simulation.
    """

    def __init__(self, agent_id, position, state='S'):
        """
        Initialize an agent.

        Parameters
        ----------
        agent_id : int
            Unique identifier
        position : tuple
            (x, y) position
        state : str
            'S', 'I', or 'R'
        """
        self.id = agent_id
        self.position = np.array(position, dtype=float)
        self.state = state
        self.infection_time = None

    def move(self, speed, domain_size):
        """
        Random walk movement.

        Parameters
        ----------
        speed : float
            Maximum movement distance per step
        domain_size : float
            Size of square domain
        """
        # Random direction
        angle = np.random.uniform(0, 2 * np.pi)
        displacement = speed * np.array([np.cos(angle), np.sin(angle)])

        # Update position
        self.position += displacement

        # Periodic boundary conditions
        self.position = self.position % domain_size

    def distance_to(self, other):
        """
        Compute distance to another agent (accounting for periodic boundaries).

        Parameters
        ----------
        other : Agent
            Another agent

        Returns
        -------
        distance : float
            Euclidean distance
        """
        return np.linalg.norm(self.position - other.position)


class ABMSimulator:
    """
    Agent-Based epidemic Model simulator.
    """

    def __init__(self, N, beta, gamma, speed, contact_radius,
                 domain_size=10.0, initial_infected=1):
        """
        Initialize ABM simulator.

        Parameters
        ----------
        N : int
            Number of agents
        beta : float
            Transmission probability upon contact
        gamma : float
            Recovery probability per time step
        speed : float
            Agent movement speed
        contact_radius : float
            Distance within which transmission can occur
        domain_size : float
            Size of square domain
        initial_infected : int
            Number of initially infected agents
        """
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.speed = speed
        self.contact_radius = contact_radius
        self.domain_size = domain_size

        # Initialize agents
        self.agents = []
        for i in range(N):
            # Random initial position
            pos = np.random.uniform(0, domain_size, size=2)
            agent = Agent(i, pos, state='S')
            self.agents.append(agent)

        # Set initial infections
        infected_indices = np.random.choice(N, size=initial_infected, replace=False)
        for idx in infected_indices:
            self.agents[idx].state = 'I'
            self.agents[idx].infection_time = 0

        # Initialize history
        self.history = {
            'S': [self._count_state('S')],
            'I': [self._count_state('I')],
            'R': [self._count_state('R')]
        }
        self.position_history = []
        self.state_history = []
        self._record_state()

        self.timestep = 0

    def _count_state(self, state):
        """Count agents in given state."""
        return sum(1 for agent in self.agents if agent.state == state)

    def _record_state(self):
        """Record current positions and states."""
        positions = np.array([agent.position for agent in self.agents])
        states = [agent.state for agent in self.agents]
        self.position_history.append(positions.copy())
        self.state_history.append(states.copy())

    def step(self):
        """
        Perform one time step.

        1. Move agents
        2. Detect contacts
        3. Transmit infection
        4. Recover agents

        Returns
        -------
        counts : dict
            S, I, R counts
        """
        self.timestep += 1

        # Move all agents
        for agent in self.agents:
            agent.move(self.speed, self.domain_size)

        # Detect contacts and transmit
        newly_infected = []
        for i, agent in enumerate(self.agents):
            if agent.state == 'I':
                # Check contacts with susceptible agents
                for j, other in enumerate(self.agents):
                    if other.state == 'S':
                        distance = agent.distance_to(other)
                        if distance < self.contact_radius:
                            # Transmission event
                            if np.random.random() < self.beta:
                                newly_infected.append(j)

        # Apply infections
        for idx in newly_infected:
            self.agents[idx].state = 'I'
            self.agents[idx].infection_time = self.timestep

        # Process recoveries
        for agent in self.agents:
            if agent.state == 'I':
                if np.random.random() < self.gamma:
                    agent.state = 'R'

        # Record state
        counts = {
            'S': self._count_state('S'),
            'I': self._count_state('I'),
            'R': self._count_state('R')
        }

        self.history['S'].append(counts['S'])
        self.history['I'].append(counts['I'])
        self.history['R'].append(counts['R'])
        self._record_state()

        return counts

    def simulate(self, max_steps=200):
        """
        Run simulation until no infections or max steps.

        Parameters
        ----------
        max_steps : int
            Maximum number of steps

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

    def plot_positions(self, timestep=None, ax=None, show_contacts=False):
        """
        Plot agent positions at a given timestep.

        Parameters
        ----------
        timestep : int, optional
            Timestep to plot (if None, plot current state)
        ax : matplotlib axes, optional
            Axes to plot on
        show_contacts : bool
            Whether to show contact radius

        Returns
        -------
        ax : matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        if timestep is None:
            positions = np.array([agent.position for agent in self.agents])
            states = [agent.state for agent in self.agents]
        else:
            positions = self.position_history[timestep]
            states = self.state_history[timestep]

        # Separate by state
        s_pos = positions[[i for i, s in enumerate(states) if s == 'S']]
        i_pos = positions[[i for i, s in enumerate(states) if s == 'I']]
        r_pos = positions[[i for i, s in enumerate(states) if s == 'R']]

        # Plot agents
        if len(s_pos) > 0:
            ax.scatter(s_pos[:, 0], s_pos[:, 1], c='#4A90E2', s=50,
                      label='Susceptible', alpha=0.7, edgecolors='black', linewidth=0.5)
        if len(i_pos) > 0:
            ax.scatter(i_pos[:, 0], i_pos[:, 1], c='#E94B3C', s=50,
                      label='Infected', alpha=0.7, edgecolors='black', linewidth=0.5)
        if len(r_pos) > 0:
            ax.scatter(r_pos[:, 0], r_pos[:, 1], c='#50C878', s=50,
                      label='Recovered', alpha=0.7, edgecolors='black', linewidth=0.5)

        # Show contact radius for one infected agent
        if show_contacts and len(i_pos) > 0:
            circle = plt.Circle(i_pos[0], self.contact_radius,
                              color='red', fill=False, linestyle='--',
                              linewidth=1.5, alpha=0.5)
            ax.add_patch(circle)

        ax.set_xlim(0, self.domain_size)
        ax.set_ylim(0, self.domain_size)
        ax.set_aspect('equal')
        ax.set_xlabel('X position', fontsize=12)
        ax.set_ylabel('Y position', fontsize=12)

        title = f'Agent Positions (t={timestep if timestep is not None else self.timestep})'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        return ax


def plot_abm_timeseries(history, title="ABM Epidemic", ax=None):
    """
    Plot S, I, R time series from ABM simulation.

    Parameters
    ----------
    history : dict
        Dictionary with 'S', 'I', 'R' keys
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


def run_multiple_abm_simulations(n_runs, N, beta, gamma, speed, contact_radius,
                                  domain_size=10.0, initial_infected=1, max_steps=200):
    """
    Run multiple ABM simulations and collect results.

    Parameters
    ----------
    n_runs : int
        Number of simulation runs
    N : int
        Number of agents
    beta : float
        Transmission probability
    gamma : float
        Recovery probability
    speed : float
        Agent movement speed
    contact_radius : float
        Contact radius for transmission
    domain_size : float
        Domain size
    initial_infected : int
        Initial infected count
    max_steps : int
        Maximum steps per simulation

    Returns
    -------
    all_histories : list
        List of history dictionaries from each run
    """
    all_histories = []

    for _ in range(n_runs):
        sim = ABMSimulator(N, beta, gamma, speed, contact_radius,
                          domain_size, initial_infected)
        history = sim.simulate(max_steps)
        all_histories.append(history)

    return all_histories


def plot_multiple_runs(all_histories, title="Multiple ABM Runs", ax=None,
                       show_individual=True, show_mean=True):
    """
    Plot results from multiple ABM runs.

    Parameters
    ----------
    all_histories : list
        List of history dictionaries
    title : str
        Plot title
    ax : matplotlib axes, optional
        Axes to plot on
    show_individual : bool
        Whether to show individual runs
    show_mean : bool
        Whether to show mean trajectory

    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Find maximum length
    max_len = max(len(h['S']) for h in all_histories)

    # Pad histories to same length (hold final value)
    def pad_history(arr, target_len):
        if len(arr) < target_len:
            return np.pad(arr, (0, target_len - len(arr)),
                         mode='edge')
        return np.array(arr)

    # Collect all runs
    S_all = np.array([pad_history(h['S'], max_len) for h in all_histories])
    I_all = np.array([pad_history(h['I'], max_len) for h in all_histories])
    R_all = np.array([pad_history(h['R'], max_len) for h in all_histories])

    t = np.arange(max_len)

    # Plot individual runs
    if show_individual:
        for i in range(len(all_histories)):
            ax.plot(t, S_all[i], color='#4A90E2', alpha=0.2, linewidth=1)
            ax.plot(t, I_all[i], color='#E94B3C', alpha=0.2, linewidth=1)
            ax.plot(t, R_all[i], color='#50C878', alpha=0.2, linewidth=1)

    # Plot mean trajectories
    if show_mean:
        S_mean = np.mean(S_all, axis=0)
        I_mean = np.mean(I_all, axis=0)
        R_mean = np.mean(R_all, axis=0)

        ax.plot(t, S_mean, label='Susceptible (mean)', linewidth=3,
               color='#4A90E2')
        ax.plot(t, I_mean, label='Infected (mean)', linewidth=3,
               color='#E94B3C')
        ax.plot(t, R_mean, label='Recovered (mean)', linewidth=3,
               color='#50C878')

    ax.set_xlabel('Time (steps)', fontsize=12)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return ax
