# example.py
"""
This script demonstrates the usage of a custom plotting style and color map in matplotlib.

The script performs the following steps:
1. Imports the necessary modules.
2. Applies a custom plotting style using the `set_plt_style` function from `script.plt_style`.
3. Provides instructions on how to use a custom color map (`custom_cmap`) from `script.plt_style`.
4. Creates a simple line plot with labeled axes and a title.

Usage:
- To apply the custom style, ensure that `set_plt_style` is defined in `script.plt_style`.
- To use the custom color map, uncomment the relevant lines, import `get_custom_cmap` from `script.plt_style`, and use it in the `plt.imshow` function.

Example:
    # random_data = np.random.rand(10, 10)

The script concludes by creating a simple line plot with labeled axes and a title.
"""
import matplotlib.pyplot as plt
from script.plt_style import set_plt_style



# Apply the custom style
set_plt_style()
'''
to use custom color map 'custom_cmap'  import it from script.plt_style.py and use it as shown below:
!!Note: you must not refer to it as cmap='custom_cmap' in the plt.imshow() function, instead use cmap=custom_cmap!!
'''
# import numpy as np
# from script.plt_style import get_custom_cmap
# custom_cmap = get_custom_cmap()
# random_data=np.random.rand(10,10)
# plt.imshow(random_data, cmap=custom_cmap)
# plt.colorbar()
# plt.show()


# Now create your plots
plt.plot([0, 1, 2, 3], [10, 20, 25, 30])
plt.xlabel('X-axis label')
plt.ylabel('Y-axis label')
plt.title('Example Plot')
plt.show()
