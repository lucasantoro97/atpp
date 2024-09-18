"""
This module provides an example of how to use custom plotting styles in matplotlib.

Functions:
    - example_plot: Create an example plot using the custom plotting style.

Example usage:
    >>> from plottingstyle_example import example_plot
    >>> example_plot()
"""

import matplotlib.pyplot as plt
from atpp.plt_style import set_plt_style, get_custom_cmap
import numpy as np

def example_plot():
    """
    Create an example plot using the custom plotting style.

    This function demonstrates how to apply the custom plotting style and color map
    to a simple plot. It generates a sine wave and plots it using the custom style.

    Example:
        >>> example_plot()
    """
    # Apply custom plotting style
    set_plt_style()

    # Generate example data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(x, y, label='Sine Wave')

    # Apply custom color map
    cmap = get_custom_cmap()
    sc = ax.scatter(x, y, c=y, cmap=cmap)

    # Add color bar
    plt.colorbar(sc, ax=ax, label='Amplitude')

    # Add labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Example Plot with Custom Style')
    ax.legend()

    # Show the plot
    plt.show()

example_plot()