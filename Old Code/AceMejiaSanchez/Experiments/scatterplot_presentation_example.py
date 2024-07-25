# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 10:56:09 2024

@author: aceme
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# PRE-AUGMENTATION
#x1,y1 = make_moons(n_samples=40, random_state=42, noise=.2)

# POST-AUGMENTATION
x1,y1 = make_moons(n_samples=1500, random_state=42, noise=.2)

# Create a scatter plot
plt.figure(figsize=(10, 6))

# Plot data points for each class
plt.scatter(x1[:,0], x1[:,1], c = y1, cmap = "coolwarm", s = 80, edgecolors='black')

# Add title and labels
plt.rcParams['font.size'] = 20
#plt.title('Pre-Augmentation')
plt.title('Post-Augmentation')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')

# Set x and y axis limits
plt.xlim(-2, 3)
plt.ylim(-2, 2)

plt.gca().set_xticks([])
plt.gca().set_yticks([])

# Save plot
#plt.savefig('C:/Users/aceme/OneDrive/Documents/GitHub/BP24/AceMejiaSanchez/Images/pre_data_aug.png', dpi=300)
plt.savefig('C:/Users/aceme/OneDrive/Documents/GitHub/BP24/AceMejiaSanchez/Images/post_data_aug.png', dpi=300)

# Show the plot
plt.show()