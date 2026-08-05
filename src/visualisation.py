"""
Visualization utilities for epidemic models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Tuple


def setup_plot_style():
    """Set up consistent plot styling."""
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['lines.linewidth'] = 2


def plot_timeseries(t, data_dict, labels=None, colors=None,
                   title="Epidemic Dynamics", xlabel="Time (days)",
                   ylabel="Population", ax=None, **kwargs):
    """
    Generic time series plotting function.

    Parameters
    ----------
    t : array
        Time array
    data_dict : dict or list of arrays
        Dictionary mapping labels to data arrays, or list of arrays
    labels : list, optional
        Labels for each series (if data_dict is a list)
    colors : list, optional
        Colors for each series
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    ax : matplotlib axes, optional
        Axes to plot on
    **kwargs : dict
        Additional arguments passed to plot()

    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Default colors
    if colors is None:
        colors = ['#4A90E2', '#E94B3C', '#50C878', '#F5A623', '#9B59B6']

    # Handle dict or list input
    if isinstance(data_dict, dict):
        items = list(data_dict.items())
    else:
        if labels is None:
            labels = [f"Series {i+1}" for i in range(len(data_dict))]
        items = list(zip(labels, data_dict))

    # Plot each series
    for i, (label, data) in enumerate(items):
        color = colors[i % len(colors)]
        ax.plot(t, data, label=label, color=color, **kwargs)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return ax


def plot_phase_portrait(S, I, title="Phase Portrait: S vs I",
                       ax=None, show_trajectory=True, **kwargs):
    """
    Plot phase portrait (S vs I).

    Parameters
    ----------
    S : array
        Susceptible population
    I : array
        Infected population
    title : str
        Plot title
    ax : matplotlib axes, optional
        Axes to plot on
    show_trajectory : bool
        Whether to show trajectory direction
    **kwargs : dict
        Additional arguments for plot

    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if show_trajectory:
        # Plot with color gradient showing time evolution
        n_points = len(S)
        colors = plt.cm.viridis(np.linspace(0, 1, n_points))

        for i in range(n_points - 1):
            ax.plot(S[i:i+2], I[i:i+2], color=colors[i], linewidth=2)

        # Mark start and end
        ax.plot(S[0], I[0], 'go', markersize=10, label='Start', zorder=5)
        ax.plot(S[-1], I[-1], 'ro', markersize=10, label='End', zorder=5)

    else:
        ax.plot(S, I, linewidth=2, color='#4A90E2', **kwargs)

    ax.set_xlabel('Susceptible (S)', fontsize=12)
    ax.set_ylabel('Infected (I)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return ax


def plot_heatmap(x_values, y_values, z_matrix, xlabel, ylabel, title,
                cmap='viridis', ax=None, colorbar_label="", vmin=None, vmax=None):
    """
    Plot 2D heatmap.

    Parameters
    ----------
    x_values : array
        X-axis values
    y_values : array
        Y-axis values
    z_matrix : 2D array
        Values to plot (shape: len(y_values) x len(x_values))
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    cmap : str
        Colormap name
    ax : matplotlib axes, optional
        Axes to plot on
    colorbar_label : str
        Label for colorbar
    vmin, vmax : float, optional
        Min and max values for color scaling

    Returns
    -------
    ax : matplotlib axes
    im : matplotlib image
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(z_matrix, aspect='auto', origin='lower',
                   extent=[x_values[0], x_values[-1],
                          y_values[0], y_values[-1]],
                   cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=12)

    return ax, im


def plot_degree_distribution(k_values, P_k, title="Degree Distribution",
                            log_scale=False, ax=None):
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
    log_scale : bool
        Whether to use log-log scale
    ax : matplotlib axes, optional
        Axes to plot on

    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if log_scale:
        # Filter out zero values for log scale
        mask = (k_values > 0) & (P_k > 0)
        ax.loglog(k_values[mask], P_k[mask], 'o-', color='#4A90E2',
                 markersize=6, linewidth=2)
    else:
        ax.bar(k_values, P_k, color='#4A90E2', alpha=0.7, edgecolor='black')

    ax.set_xlabel('Degree k', fontsize=12)
    ax.set_ylabel('P(k)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    return ax


def plot_network(graph, node_colors, pos=None, node_size=100,
                title="Network", ax=None, colormap=None, legend_labels=None):
    """
    Plot network graph.

    Parameters
    ----------
    graph : networkx.Graph
        Network to plot
    node_colors : list or array
        Colors for each node
    pos : dict, optional
        Node positions
    node_size : int
        Size of nodes
    title : str
        Plot title
    ax : matplotlib axes, optional
        Axes to plot on
    colormap : dict, optional
        Mapping from color values to RGB colors
    legend_labels : dict, optional
        Labels for legend

    Returns
    -------
    ax : matplotlib axes
    pos : dict
        Node positions (for reuse)
    """
    import networkx as nx

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if pos is None:
        pos = nx.spring_layout(graph, seed=42)

    # Draw edges
    nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                          node_size=node_size, ax=ax)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    if legend_labels:
        # Create legend patches
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=label)
                          for label, color in legend_labels.items()]
        ax.legend(handles=legend_elements, fontsize=11)

    return ax, pos


def plot_comparison(data_list, labels, t=None, title="Model Comparison",
                   xlabel="Time (days)", ylabel="Infected Population",
                   colors=None, ax=None):
    """
    Plot multiple model results for comparison.

    Parameters
    ----------
    data_list : list of arrays
        Data from different models
    labels : list of str
        Labels for each model
    t : array, optional
        Time array (if None, use indices)
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    colors : list, optional
        Colors for each series
    ax : matplotlib axes, optional
        Axes to plot on

    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if t is None:
        t = np.arange(len(data_list[0]))

    if colors is None:
        colors = ['#4A90E2', '#E94B3C', '#50C878', '#F5A623', '#9B59B6']

    for i, (data, label) in enumerate(zip(data_list, labels)):
        color = colors[i % len(colors)]
        ax.plot(t, data, label=label, color=color, linewidth=2)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    return ax


def save_figure(fig, filename, output_dir="../assets/figures",
               dpi=300, formats=['png']):
    """
    Save figure to file.

    Parameters
    ----------
    fig : matplotlib figure
        Figure to save
    filename : str
        Base filename (without extension)
    output_dir : str
        Output directory
    dpi : int
        Resolution
    formats : list
        List of file formats to save
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    for fmt in formats:
        filepath = os.path.join(output_dir, f"{filename}.{fmt}")
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {filepath}")


# Apply default style on import
setup_plot_style()
