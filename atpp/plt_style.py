    """
    This module provides functions to set custom matplotlib styles and create custom color maps.
    
    Functions:
        - set_plt_style: Set custom matplotlib style parameters.
        - get_custom_cmap: Create and return a custom color map.
    
    Example usage:
        >>> from plt_style import set_plt_style, get_custom_cmap
        >>> set_plt_style()
        >>> cmap = get_custom_cmap()
    """
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    
    def set_plt_style():
        """
        Set custom matplotlib style parameters for consistent plotting aesthetics.
    
        This function sets various matplotlib rcParams to customize the appearance of plots,
        including figure size, font family, font size, line width, tick size, legend properties,
        and more.
    
        Example:
            >>> set_plt_style()
        """
        # Set figure size
        width = 3
        height = width * 0.8
        plt.rcParams['figure.figsize'] = (width, height)
        
        # Set matplotlib global rcParams
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.linewidth'] = 2
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['lines.markersize'] = 10
        plt.rcParams['xtick.major.size'] = 8
        plt.rcParams['xtick.major.width'] = 2
        plt.rcParams['ytick.major.size'] = 8
        plt.rcParams['ytick.major.width'] = 2
        plt.rcParams['xtick.minor.size'] = 4
        plt.rcParams['ytick.minor.size'] = 4
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['legend.frameon'] = True
        plt.rcParams['legend.framealpha'] = 1
        plt.rcParams['figure.subplot.left'] = 0.2
        plt.rcParams['figure.subplot.right'] = 0.9
        plt.rcParams['figure.subplot.bottom'] = 0.2
        plt.rcParams['figure.subplot.top'] = 0.88
        plt.rcParams['axes.formatter.useoffset'] = True
        plt.rcParams['axes.formatter.offset_threshold'] = 2
        plt.rcParams['axes.grid'] = False
        plt.rcParams['xtick.minor.visible'] = True
        plt.rcParams['ytick.minor.visible'] = True
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['axes.facecolor'] = 'whitesmoke'
        plt.rcParams['axes.labelpad'] = 5
        plt.rcParams['legend.facecolor'] = 'white'
    
    def get_custom_cmap():
        """
        Create and return a custom color map for use in matplotlib plots.
    
        This function creates a custom color map that transitions through white, blue, green, and red.
        The color map is discretized into 100 bins.
    
        :return: A custom LinearSegmentedColormap object.
        :rtype: matplotlib.colors.LinearSegmentedColormap
    
        Example:
            >>> cmap = get_custom_cmap()
            >>> plt.imshow(data, cmap=cmap)
        """
        # Custom color map
        colors = [(1, 1, 1), (49/255, 86/255, 177/255), (174/255, 201/255, 65/255), (1, 0, 0)]
        n_bins = 100  # Discretizes the interpolation into bins
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=n_bins)
        return custom_cmap