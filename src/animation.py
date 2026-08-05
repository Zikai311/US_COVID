"""
Animation utilities for epidemic models.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import networkx as nx
from typing import Optional, Callable


def animate_network_spread(network_sim, pos=None, interval=200,
                          filename=None, output_dir="../assets/gifs",
                          fps=5, node_size=100):
    """
    Animate epidemic spread on a network.

    Parameters
    ----------
    network_sim : NetworkSIR
        NetworkSIR simulator object with completed simulation
    pos : dict, optional
        Node positions (if None, computed with spring layout)
    interval : int
        Delay between frames in milliseconds
    filename : str, optional
        Output filename (without extension)
    output_dir : str
        Output directory
    fps : int
        Frames per second for saved animation
    node_size : int
        Size of network nodes

    Returns
    -------
    anim : FuncAnimation
        Animation object
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    graph = network_sim.graph
    if pos is None:
        pos = nx.spring_layout(graph, seed=42)

    # Get history length
    n_steps = len(network_sim.history['S'])

    # Color mapping
    color_map = {'S': '#4A90E2', 'I': '#E94B3C', 'R': '#50C878'}

    def init():
        ax.clear()
        ax.set_title('Network Epidemic Spread - t=0', fontsize=14, fontweight='bold')
        ax.axis('off')
        return []

    def update(frame):
        ax.clear()

        # Set node states to match this frame
        # Note: This requires storing state history during simulation
        # For now, we'll animate the final state
        node_colors = [color_map[graph.nodes[node]['state']]
                      for node in graph.nodes()]

        # Draw network
        nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax)
        nx.draw_networkx_nodes(graph, pos, node_color=node_colors,
                              node_size=node_size, ax=ax)

        # Add title with counts
        S_count = network_sim.history['S'][min(frame, n_steps-1)]
        I_count = network_sim.history['I'][min(frame, n_steps-1)]
        R_count = network_sim.history['R'][min(frame, n_steps-1)]

        title = f'Network Epidemic Spread - t={frame}\n'
        title += f'S: {S_count}, I: {I_count}, R: {R_count}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map['S'], label='Susceptible'),
            Patch(facecolor=color_map['I'], label='Infected'),
            Patch(facecolor=color_map['R'], label='Recovered')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

        return []

    anim = FuncAnimation(fig, update, init_func=init,
                        frames=n_steps, interval=interval,
                        blit=False, repeat=True)

    if filename:
        save_animation(anim, filename, output_dir, fps)

    return anim


def animate_agent_motion(abm_sim, interval=100, filename=None,
                        output_dir="../assets/gifs", fps=10,
                        show_contacts=False):
    """
    Animate agent-based model with moving agents.

    Parameters
    ----------
    abm_sim : ABMSimulator
        ABM simulator with completed simulation
    interval : int
        Delay between frames in milliseconds
    filename : str, optional
        Output filename
    output_dir : str
        Output directory
    fps : int
        Frames per second
    show_contacts : bool
        Whether to show contact radius

    Returns
    -------
    anim : FuncAnimation
        Animation object
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    n_frames = len(abm_sim.position_history)
    color_map = {'S': '#4A90E2', 'I': '#E94B3C', 'R': '#50C878'}

    def init():
        ax.clear()
        ax.set_xlim(0, abm_sim.domain_size)
        ax.set_ylim(0, abm_sim.domain_size)
        ax.set_aspect('equal')
        ax.set_xlabel('X position', fontsize=12)
        ax.set_ylabel('Y position', fontsize=12)
        ax.grid(True, alpha=0.3)
        return []

    def update(frame):
        ax.clear()

        positions = abm_sim.position_history[frame]
        states = abm_sim.state_history[frame]

        # Separate by state
        s_pos = positions[[i for i, s in enumerate(states) if s == 'S']]
        i_pos = positions[[i for i, s in enumerate(states) if s == 'I']]
        r_pos = positions[[i for i, s in enumerate(states) if s == 'R']]

        # Plot agents
        if len(s_pos) > 0:
            ax.scatter(s_pos[:, 0], s_pos[:, 1], c=color_map['S'], s=50,
                      label='Susceptible', alpha=0.7, edgecolors='black', linewidth=0.5)
        if len(i_pos) > 0:
            ax.scatter(i_pos[:, 0], i_pos[:, 1], c=color_map['I'], s=50,
                      label='Infected', alpha=0.7, edgecolors='black', linewidth=0.5)
        if len(r_pos) > 0:
            ax.scatter(r_pos[:, 0], r_pos[:, 1], c=color_map['R'], s=50,
                      label='Recovered', alpha=0.7, edgecolors='black', linewidth=0.5)

        # Show contact radius for infected agents
        if show_contacts and len(i_pos) > 0:
            for pos in i_pos[:min(3, len(i_pos))]:  # Show up to 3
                circle = plt.Circle(pos, abm_sim.contact_radius,
                                  color='red', fill=False, linestyle='--',
                                  linewidth=1, alpha=0.3)
                ax.add_patch(circle)

        ax.set_xlim(0, abm_sim.domain_size)
        ax.set_ylim(0, abm_sim.domain_size)
        ax.set_aspect('equal')
        ax.set_xlabel('X position', fontsize=12)
        ax.set_ylabel('Y position', fontsize=12)

        # Title with counts
        S_count = abm_sim.history['S'][frame]
        I_count = abm_sim.history['I'][frame]
        R_count = abm_sim.history['R'][frame]
        title = f'ABM Epidemic - t={frame}\nS: {S_count}, I: {I_count}, R: {R_count}'
        ax.set_title(title, fontsize=14, fontweight='bold')

        ax.legend(fontsize=11, loc='upper right')
        ax.grid(True, alpha=0.3)

        return []

    anim = FuncAnimation(fig, update, init_func=init,
                        frames=n_frames, interval=interval,
                        blit=False, repeat=True)

    if filename:
        save_animation(anim, filename, output_dir, fps)

    return anim


def animate_epidemic_growth(t, S, I, R, interval=50, filename=None,
                           output_dir="../assets/gifs", fps=20):
    """
    Animate epidemic curve growth over time.

    Parameters
    ----------
    t : array
        Time array
    S, I, R : arrays
        Susceptible, Infected, Recovered populations
    interval : int
        Delay between frames in milliseconds
    filename : str, optional
        Output filename
    output_dir : str
        Output directory
    fps : int
        Frames per second

    Returns
    -------
    anim : FuncAnimation
        Animation object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    n_frames = len(t)

    def init():
        ax.clear()
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0, max(S.max(), I.max(), R.max()) * 1.1)
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Population', fontsize=12)
        ax.set_title('Epidemic Dynamics', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        return []

    def update(frame):
        ax.clear()

        # Plot up to current frame
        ax.plot(t[:frame+1], S[:frame+1], label='Susceptible',
               linewidth=2, color='#4A90E2')
        ax.plot(t[:frame+1], I[:frame+1], label='Infected',
               linewidth=2, color='#E94B3C')
        ax.plot(t[:frame+1], R[:frame+1], label='Recovered',
               linewidth=2, color='#50C878')

        # Mark current point
        ax.plot(t[frame], S[frame], 'o', color='#4A90E2', markersize=8)
        ax.plot(t[frame], I[frame], 'o', color='#E94B3C', markersize=8)
        ax.plot(t[frame], R[frame], 'o', color='#50C878', markersize=8)

        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0, max(S.max(), I.max(), R.max()) * 1.1)
        ax.set_xlabel('Time (days)', fontsize=12)
        ax.set_ylabel('Population', fontsize=12)
        ax.set_title(f'Epidemic Dynamics (t={t[frame]:.1f})',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        return []

    anim = FuncAnimation(fig, update, init_func=init,
                        frames=n_frames, interval=interval,
                        blit=False, repeat=True)

    if filename:
        save_animation(anim, filename, output_dir, fps)

    return anim


def save_animation(anim, filename, output_dir="../assets/gifs",
                  fps=10, file_format='gif', dpi=100):
    """
    Save animation to file.

    Parameters
    ----------
    anim : FuncAnimation
        Animation object
    filename : str
        Output filename (without extension)
    output_dir : str
        Output directory
    fps : int
        Frames per second
    file_format : str
        'gif' or 'mp4'
    dpi : int
        Resolution

    Returns
    -------
    filepath : str
        Path to saved file
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, f"{filename}.{file_format}")

    if file_format == 'gif':
        writer = PillowWriter(fps=fps)
        anim.save(filepath, writer=writer, dpi=dpi)
    elif file_format == 'mp4':
        output_dir_mp4 = output_dir.replace('gifs', 'videos')
        os.makedirs(output_dir_mp4, exist_ok=True)
        filepath = os.path.join(output_dir_mp4, f"{filename}.{file_format}")
        writer = FFMpegWriter(fps=fps)
        anim.save(filepath, writer=writer, dpi=dpi)
    else:
        raise ValueError(f"Unsupported format: {file_format}")

    print(f"Animation saved: {filepath}")
    return filepath
