# example.py
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
