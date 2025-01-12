import matplotlib.pyplot as plt
import matplotlib as mpl

def set_publication_style():
    """Set publication-quality plot style with Times New Roman and A4 dimensions."""
    # A4 size in inches
    A4_WIDTH_INCHES = 8.27
    A4_HEIGHT_INCHES = 11.69
    
    plt.style.use('default')
    
    mpl.rcParams.update({
        # Font settings
        'font.family': 'Times New Roman',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        
        # Figure settings for A4 paper
        'figure.figsize': (A4_WIDTH_INCHES * 0.9, A4_WIDTH_INCHES * 0.6),
        'figure.dpi': 300,
        
        # Grid settings
        'grid.linewidth': 0.5,
        'grid.alpha': 0.5,
        
        # Legend settings
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        
        # Line settings
        'lines.linewidth': 1,
        'lines.markersize': 4,
        
        # Axis settings
        'axes.linewidth': 0.5,
        'axes.grid': True,
        
        # Export settings
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })
