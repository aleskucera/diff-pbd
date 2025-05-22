import json
import os
from typing import List
from typing import Union

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from pbd_torch.model import Model
from pbd_torch.model import State
from pbd_torch.transform import rotate_vector_inverse


def save_simulation(model: Model, states: List[State], output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    simulation = {
        "model": model.serialize(),
        "states": [state.serialize(model) for state in states],
    }

    json_simulation = json.dumps(simulation, indent=2)
    with open(output_file, "w") as f:
        f.write(json_simulation)


def plot_points(ax: Axes, points: torch.Tensor, title: str):
    # Plot the collision points
    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        c=points[:, 2],
        marker="o",
        s=10,
        cmap="viridis",
        alpha=0.7,
    )

    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)


def plot_frame(
    ax: Axes,
    frame: torch.Tensor,
    origin_color: Union[str, tuple],
    scale: float = 1.0,
    label: str = None,
):
    origin = frame[:3]

    # Define axes
    x_axis = rotate_vector_inverse(torch.tensor([1.0, 0.0, 0.0]), frame[3:]) * scale
    y_axis = rotate_vector_inverse(torch.tensor([0.0, 1.0, 0.0]), frame[3:]) * scale
    z_axis = rotate_vector_inverse(torch.tensor([0.0, 0.0, 1.0]), frame[3:]) * scale

    # Plot axes
    ax.quiver(*origin, *x_axis, color="r")
    ax.quiver(*origin, *y_axis, color="g")
    ax.quiver(*origin, *z_axis, color="b")

    # Plot origin
    ax.scatter(*origin, color=origin_color, label=label)


def create_scientific_plot(
        time_data,
        plot_data,
        config=None,
        save_path=None
):
    """
    Universal function for creating scientific plots with high customization.

    Parameters:
    -----------
    time_data : array-like
        The time values for the x-axis

    plot_data : dict
        Dictionary containing data series to plot with structure:
        {
            'series_name': {
                'data': array-like values,
                'label': str (optional, defaults to series_name),
                'color': str (optional),
                'linestyle': str (optional),
                'linewidth': float (optional),
                'marker': str (optional),
                'markersize': float (optional),
                'alpha': float (optional),
                'zorder': int (optional),
                'axis': int or str (optional, for specifying which y-axis to use)
            }
        }

    config : dict
        Dictionary with plot configuration settings (see default_config below)

    save_path : str, optional
        Path to save the resulting figure

    Returns:
    --------
    fig, axes : matplotlib Figure and Axes objects
    """
    # Default configuration settings
    default_config = {
        'plots_to_show': None,  # If None, show all data series
        'plot_arrangement': 'horizontal',  # 'horizontal', 'vertical', or 'grid'
        'grid_shape': None,  # For 'grid' arrangement, specify (rows, cols)
        'figsize': (12, 5),
        'dpi': 300,
        'title': None,
        'suptitle': None,
        'shared_x': True,
        'shared_y': False,
        'x_label': 'Time (s)',
        'y_labels': {},  # Dict mapping series name to y-axis label
        'plot_titles': {},  # Dict mapping series name to subplot title
        'y_limits': {},  # Dict mapping series name to (min, max) y-limits
        'font_sizes': {
            'title': 14,
            'suptitle': 16,
            'axis_label': 12,
            'tick_label': 10,
            'legend': 11
        },
        'grid': {
            'show': True,
            'linestyle': '--',
            'alpha': 0.7
        },
        'legend': {
            'show': True,
            'location': 'best',
            'frameon': True,
            'framealpha': 0.8
        },
        'colors': {},  # Dict mapping series name to color
        'line_styles': {},  # Dict mapping series name to line style
        'line_widths': {},  # Dict mapping series name to line width
        'markers': {},  # Dict mapping series name to marker style
        'marker_sizes': {},  # Dict mapping series name to marker size
        'tight_layout': True,
        'tight_layout_params': {'rect': [0, 0, 1, 0.96]},
        'subplots_adjust': None,  # Dict with subplots_adjust parameters
        'spines': {  # Control visibility of plot borders
            'top': False,
            'right': False,
            'left': True,
            'bottom': True
        },
        'tick_direction': 'out',  # 'in', 'out', or 'inout'
        'minor_ticks': True,
        'x_grid_minor': False,
        'y_grid_minor': False,
        'background_color': 'white'
    }

    # Update default config with user-provided config
    if config is not None:
        for key, value in config.items():
            if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                default_config[key].update(value)
            else:
                default_config[key] = value

    config = default_config

    # Determine which plots to show
    if config['plots_to_show'] is None:
        plots_to_show = list(plot_data.keys())
    else:
        plots_to_show = [p for p in config['plots_to_show'] if p in plot_data]

    n_plots = len(plots_to_show)

    # Set up figure and axes based on arrangement
    if config['plot_arrangement'] == 'horizontal':
        fig, axes = plt.subplots(1, n_plots, figsize=config['figsize'],
                                 sharex=config['shared_x'], sharey=config['shared_y'])
    elif config['plot_arrangement'] == 'vertical':
        fig, axes = plt.subplots(n_plots, 1, figsize=config['figsize'],
                                 sharex=config['shared_x'], sharey=config['shared_y'])
    elif config['plot_arrangement'] == 'grid':
        if config['grid_shape'] is None:
            # Calculate a reasonable grid shape
            import math
            cols = math.ceil(math.sqrt(n_plots))
            rows = math.ceil(n_plots / cols)
        else:
            rows, cols = config['grid_shape']

        fig, axes = plt.subplots(rows, cols, figsize=config['figsize'],
                                 sharex=config['shared_x'], sharey=config['shared_y'])
    else:
        raise ValueError(f"Invalid plot_arrangement: {config['plot_arrangement']}")

    # Set figure background color
    fig.patch.set_facecolor(config['background_color'])

    # Handle case of single subplot
    if n_plots == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes.flatten()
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Add the main suptitle if specified
    if config['suptitle']:
        fig.suptitle(config['suptitle'], fontsize=config['font_sizes']['suptitle'])

    # Create each plot
    for i, plot_key in enumerate(plots_to_show):
        if i >= len(axes):  # Stop if we run out of axes
            break

        ax = axes[i]

        # Get plot data and metadata
        plot_info = plot_data[plot_key]
        if not isinstance(plot_info, dict):  # If just the data array is provided
            plot_info = {'data': plot_info}

        # Set plot styles with fallbacks to defaults
        label = plot_info.get('label', plot_key)
        color = plot_info.get('color', config['colors'].get(plot_key, None))
        linestyle = plot_info.get('linestyle', config['line_styles'].get(plot_key, '-'))
        linewidth = plot_info.get('linewidth', config['line_widths'].get(plot_key, 1.5))
        marker = plot_info.get('marker', config['markers'].get(plot_key, None))
        markersize = plot_info.get('markersize', config['marker_sizes'].get(plot_key, 6))
        alpha = plot_info.get('alpha', 1.0)
        zorder = plot_info.get('zorder', 5)

        # Create the plot
        ax.plot(
            time_data,
            plot_info['data'],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            marker=marker,
            markersize=markersize,
            alpha=alpha,
            zorder=zorder
        )

        # Set title if specified
        title = config['plot_titles'].get(plot_key, plot_key if config['title'] is None else config['title'])
        ax.set_title(title, fontsize=config['font_sizes']['title'])

        # Set x and y labels
        if i == 0 or not config['shared_y']:  # Only add y label for first plot or if not shared
            y_label = config['y_labels'].get(plot_key, '')
            ax.set_ylabel(y_label, fontsize=config['font_sizes']['axis_label'])

        # Only add x label for the bottom plots or if not shared
        if (config['plot_arrangement'] == 'horizontal' or
                i == n_plots - 1 or
                not config['shared_x']):
            ax.set_xlabel(config['x_label'], fontsize=config['font_sizes']['axis_label'])

        # Set y limits if specified
        if plot_key in config['y_limits']:
            ax.set_ylim(config['y_limits'][plot_key])

        # Configure ticks
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=config['font_sizes']['tick_label'],
            direction=config['tick_direction']
        )

        # Configure spines
        for spine, visible in config['spines'].items():
            ax.spines[spine].set_visible(visible)

        # Configure grid
        if config['grid']['show']:
            ax.grid(True, linestyle=config['grid']['linestyle'], alpha=config['grid']['alpha'])
            if config['minor_ticks']:
                ax.minorticks_on()
                if config['x_grid_minor']:
                    ax.grid(True, which='minor', axis='x', alpha=config['grid']['alpha'] * 0.5)
                if config['y_grid_minor']:
                    ax.grid(True, which='minor', axis='y', alpha=config['grid']['alpha'] * 0.5)

        # Add legend if specified
        if config['legend']['show']:
            ax.legend(
                loc=config['legend']['location'],
                fontsize=config['font_sizes']['legend'],
                frameon=config['legend']['frameon'],
                framealpha=config['legend']['framealpha']
            )

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Adjust layout
    if config['tight_layout']:
        plt.tight_layout(**config['tight_layout_params'])

    if config['subplots_adjust'] is not None:
        plt.subplots_adjust(**config['subplots_adjust'])

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight')

    return fig, axes


def create_scientific_subplot_plot(time_data, plot_data, config=None, save_path=None):
    """
    Create scientific plots with multiple data series per subplot.

    Parameters:
    -----------
    time_data : array-like
        The time values for the x-axis

    plot_data : dict
        Dictionary with structure:
        {
            'subplot_name': {
                'series_name': {
                    'data': array-like values,
                    'label': str,
                    'color': str,
                    'linestyle': str,
                    'linewidth': float,
                    ...
                },
                'another_series': {...}
            },
            'another_subplot': {...}
        }

    config : dict
        Dictionary with plot configuration settings

    save_path : str, optional
        Path to save the resulting figure

    Returns:
    --------
    fig, axes : matplotlib Figure and Axes objects
    """
    # Default configuration settings
    default_config = {
        'plots_to_show': None,  # If None, show all subplots
        'plot_arrangement': 'horizontal',  # 'horizontal', 'vertical', or 'grid'
        'grid_shape': None,  # For 'grid' arrangement, specify (rows, cols)
        'figsize': (12, 5),
        'dpi': 300,
        'title': None,
        'suptitle': None,
        'shared_x': True,
        'shared_y': False,
        'x_label': 'Time (s)',
        'y_labels': {},  # Dict mapping subplot name to y-axis label
        'plot_titles': {},  # Dict mapping subplot name to subplot title
        'y_limits': {},  # Dict mapping subplot name to (min, max) y-limits
        'font_sizes': {
            'title': 14,
            'suptitle': 16,
            'axis_label': 12,
            'tick_label': 10,
            'legend': 11
        },
        'grid': {
            'show': True,
            'linestyle': '--',
            'alpha': 0.7
        },
        'legend': {
            'show': True,
            'location': 'best',
            'frameon': True,
            'framealpha': 0.8
        },
        'tight_layout': True,
        'tight_layout_params': {'rect': [0, 0, 1, 0.96]},
        'subplots_adjust': None,  # Dict with subplots_adjust parameters
        'spines': {  # Control visibility of plot borders
            'top': False,
            'right': False,
            'left': True,
            'bottom': True
        },
        'tick_direction': 'out',  # 'in', 'out', or 'inout'
        'minor_ticks': True,
        'x_grid_minor': False,
        'y_grid_minor': False,
        'background_color': 'white'
    }

    # Update default config with user-provided config
    if config is not None:
        for key, value in config.items():
            if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                default_config[key].update(value)
            else:
                default_config[key] = value

    config = default_config

    # Determine which plots to show
    if config['plots_to_show'] is None:
        plots_to_show = list(plot_data.keys())
    else:
        plots_to_show = [p for p in config['plots_to_show'] if p in plot_data]

    n_plots = len(plots_to_show)

    # Set up figure and axes based on arrangement
    if config['plot_arrangement'] == 'horizontal':
        fig, axes = plt.subplots(1, n_plots, figsize=config['figsize'],
                                 sharex=config['shared_x'], sharey=config['shared_y'])
    elif config['plot_arrangement'] == 'vertical':
        fig, axes = plt.subplots(n_plots, 1, figsize=config['figsize'],
                                 sharex=config['shared_x'], sharey=config['shared_y'])
    elif config['plot_arrangement'] == 'grid':
        if config['grid_shape'] is None:
            # Calculate a reasonable grid shape
            import math
            cols = math.ceil(math.sqrt(n_plots))
            rows = math.ceil(n_plots / cols)
        else:
            rows, cols = config['grid_shape']

        fig, axes = plt.subplots(rows, cols, figsize=config['figsize'],
                                 sharex=config['shared_x'], sharey=config['shared_y'])
    else:
        raise ValueError(f"Invalid plot_arrangement: {config['plot_arrangement']}")

    # Set figure background color
    fig.patch.set_facecolor(config['background_color'])

    # Handle case of single subplot
    if n_plots == 1:
        axes = [axes] if not isinstance(axes, np.ndarray) else axes.flatten()
    else:
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    # Add the main suptitle if specified
    if config['suptitle']:
        fig.suptitle(config['suptitle'], fontsize=config['font_sizes']['suptitle'])

    # Create each subplot
    for i, subplot_key in enumerate(plots_to_show):
        if i >= len(axes):  # Stop if we run out of axes
            break

        ax = axes[i]

        # Get data series for this subplot
        subplot_data = plot_data[subplot_key]

        # Special case for 'f' which might have a direct data entry
        if subplot_key == 'f' and 'data' in subplot_data:
            # This is a single series subplot
            series_data = {subplot_key: subplot_data}

            for series_key, series_info in series_data.items():
                ax.plot(
                    time_data,
                    series_info['data'],
                    label=series_info.get('label', series_key),
                    color=series_info.get('color', None),
                    linestyle=series_info.get('linestyle', '-'),
                    linewidth=series_info.get('linewidth', 1.5),
                    marker=series_info.get('marker', None),
                    markersize=series_info.get('markersize', 6),
                    alpha=series_info.get('alpha', 1.0),
                    zorder=series_info.get('zorder', 5)
                )
        else:
            # Plot each series in this subplot
            for series_key, series_info in subplot_data.items():
                ax.plot(
                    time_data,
                    series_info['data'],
                    label=series_info.get('label', series_key),
                    color=series_info.get('color', None),
                    linestyle=series_info.get('linestyle', '-'),
                    linewidth=series_info.get('linewidth', 1.5),
                    marker=series_info.get('marker', None),
                    markersize=series_info.get('markersize', 6),
                    alpha=series_info.get('alpha', 1.0),
                    zorder=series_info.get('zorder', 5)
                )

        # Set title if specified
        title = config['plot_titles'].get(subplot_key, subplot_key if config['title'] is None else config['title'])
        ax.set_title(title, fontsize=config['font_sizes']['title'])

        # Set x and y labels
        if i == 0 or not config['shared_y']:  # Only add y label for first plot or if not shared
            y_label = config['y_labels'].get(subplot_key, '')
            ax.set_ylabel(y_label, fontsize=config['font_sizes']['axis_label'])

        # Only add x label for the bottom plots or if not shared
        if (config['plot_arrangement'] == 'horizontal' or
                i == n_plots - 1 or
                not config['shared_x']):
            ax.set_xlabel(config['x_label'], fontsize=config['font_sizes']['axis_label'])

        # Set y limits if specified
        if subplot_key in config['y_limits']:
            ax.set_ylim(config['y_limits'][subplot_key])

        # Configure ticks
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=config['font_sizes']['tick_label'],
            direction=config['tick_direction']
        )

        # Configure spines
        for spine, visible in config['spines'].items():
            ax.spines[spine].set_visible(visible)

        # Configure grid
        if config['grid']['show']:
            ax.grid(True, linestyle=config['grid']['linestyle'], alpha=config['grid']['alpha'])
            if config['minor_ticks']:
                ax.minorticks_on()
                if config['x_grid_minor']:
                    ax.grid(True, which='minor', axis='x', alpha=config['grid']['alpha'] * 0.5)
                if config['y_grid_minor']:
                    ax.grid(True, which='minor', axis='y', alpha=config['grid']['alpha'] * 0.5)

        # Add legend if specified
        if config['legend']['show']:
            ax.legend(
                loc=config['legend']['location'],
                fontsize=config['font_sizes']['legend'],
                frameon=config['legend']['frameon'],
                framealpha=config['legend']['framealpha']
            )

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # Adjust layout
    if config['tight_layout']:
        plt.tight_layout(**config['tight_layout_params'])

    if config['subplots_adjust'] is not None:
        plt.subplots_adjust(**config['subplots_adjust'])

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=config['dpi'], bbox_inches='tight')

    return fig, axes


