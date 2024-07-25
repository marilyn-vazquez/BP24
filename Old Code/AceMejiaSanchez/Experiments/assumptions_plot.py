# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 14:11:57 2024

@author: aceme
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
################## Import data ######################################################
df = pd.read_csv("C:/Users/aceme/OneDrive/Documents/SIAM Simons Summer Opportunity/Datasets/WVS_Cross-National_Wave_7_csv_v6_0.csv")

# Variables
x1 = df['lifeexpectHDI']
y1 = df['incomeindexHDI'] 

# incomeindexHDI: Income Index (0 to 1) [UNDP, 2018]
# lifeexpectHDI: 

# Create a scatter plot
plt.figure(figsize=(10, 6))

# Plot data points for each class
plt.scatter(x1, y1, s=30, edgecolors='black')

# Add title and labels
plt.rcParams['font.size'] = 20
plt.title('Life Expectancy Index v. Income Index')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')

# Set x and y axis limits
plt.xlim(0, 1)
plt.ylim(0, 1)

# plt.gca().set_xticks([])
# plt.gca().set_yticks([])

# Save plot
plt.savefig('C:/Users/aceme/OneDrive/Documents/GitHub/BP24/AceMejiaSanchez/Images/assumptions_plot.png', dpi=300)

# Show the plot
plt.show()

