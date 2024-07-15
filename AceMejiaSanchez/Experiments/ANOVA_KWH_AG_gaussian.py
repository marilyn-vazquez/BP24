# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:53:52 2024

@author: aceme
"""
################## Import relevant packages
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import random
from scipy.stats import f_oneway
from itertools import combinations
from scipy.stats import kruskal
from scipy.stats import alexandergovern

################## Import function ############################################
import sys
sys.path.append('C:/Users/aceme/OneDrive/Documents/GitHub/BP24/')
import Fake_Patterns

################## Import Data & X_train ######################################

# Import data
df = pd.read_csv("C:/Users/aceme/OneDrive/Documents/GitHub/BP24/Ellee/Data/Stacked/stacked_orig.csv")
# Indexing through pre-prepared splitting in stacked_all
X_train = df.iloc[:168, :8] # Gaussian (DO NOT CHANGE)
# X_test = df.iloc[168:241, :8]
# y_train =  df.iloc[:168, 22]
# y_test = df.iloc[168:241, 22]

################## ANOVA #########################################

print("\n------------------ ANOVA (Feature vs Label) -----------------------")

# Finds dependency between all features in X_train & the label in y_train
def anova_fvl(X_train, y_train):
    
    # Combining X_train and y_train
    df = X_train
    df['y_train'] = y_train

    # Number of features, excluding label
    var_count = len(X_train.columns)-1

    # Creates an empty array for f-statistic and P-values
    # results = []
    
    
    # TEMPORARY: Creates an empty array for tracking SIGNIFICANT counts
    siggies = []

    for i in range(0, var_count):
        
        # Compute ANOVA
        f_statistic, p_value = f_oneway(df.iloc[:,i], df.iloc[:,-1])
        
        # TEMPORARY: Save p-value significance into list
        if p_value < 0.05:
            siggies.append(True)
        else:
            siggies.append(False)
    
    return siggies

# Testing consistency of ANOVA test

ANOVA_sigs = []

for i in range(16, 25, 1):
    # Loop through Poisson-distributed categorical features in stacked_all
    y_train = df.iloc[:168, i]
    
    # Run ANOVA & keep track of Poisson label & significants list
    ANOVA_sigs.append({
        "Poisson Label": i,
        "Significants List": anova_fvl(X_train, y_train)})
    
from pprint import pprint
pprint(ANOVA_sigs)

# Extract and flatten the 'Significants List' values
ANOVA_flat_list = [value for item in ANOVA_sigs for value in item['Significants List']]

# Sum the flattened list
total_sum = sum(ANOVA_flat_list)

# Count the number of True and False values
ANOVA_num_true = sum(ANOVA_flat_list)
ANOVA_num_false = len(ANOVA_flat_list) - ANOVA_num_true

print("Number of True values:", ANOVA_num_true)
print("Number of False values:", ANOVA_num_false)

################## KRUSKAL-WALLIS H Test (FvL) #######################################

print("\n------------------ Kruskal-Wallis H Test (Feature vs Label) -----------------------")

# Finds dependency between all features in X_train & the label in y_train
def kruskal_fvl(X_train, y_train):
    
    # Combining X_train and y_train
    df = X_train
    df['y_train'] = y_train

    # Number of features, excluding label
    var_count = len(X_train.columns)-1
    
    # TEMPORARY: Creates an empty array for tracking SIGNIFICANT counts
    siggies = []

    for i in range(0, var_count):
        
        # Compute KRUSKA-WALLIS H Test
        kruskal_statistic, p_value = kruskal(df.iloc[:,i], df.iloc[:,-1])
                                
        # TEMPORARY: Save p-value significance into list
        if p_value < 0.05:
            siggies.append(True)
        else:
            siggies.append(False)
    return siggies

# Testing consistency of KRUSKAL test

KRUSKAL_sigs = []

for i in range(16, 25, 1):
    # Loop through Poisson-distributed categorical features in stacked_all
    y_train = df.iloc[:168, i]
    
    # Run ANOVA
    KRUSKAL_sigs.append({
        "Poisson Label": i,
        "Significants List": kruskal_fvl(X_train, y_train)})
   
from pprint import pprint
pprint(KRUSKAL_sigs)

# Flatten the list of lists
KRUSKAL_flat_list = [value for item in KRUSKAL_sigs for value in item['Significants List']]

# Count the number of True and False values
KRUSKAL_num_true = sum(KRUSKAL_flat_list)
KRUSKAL_num_false = len(KRUSKAL_flat_list) - KRUSKAL_num_true

print("Number of True values:", KRUSKAL_num_true)
print("Number of False values:", KRUSKAL_num_false)

################## ALEXANDER-GOVERN Test #######################################

print("\n------------------ ALEXANDER-GOVERN Test (Feature vs Label) -----------------------")

# Finds dependency between all features in X_train & the label in y_train
def alexandergovern_fvl(X_train, y_train):
    
    # Combining X_train and y_train
    df = X_train
    df['y_train'] = y_train

    # Number of features, excluding label
    var_count = len(X_train.columns)-1
    
    # TEMPORARY: Creates an empty array for tracking SIGNIFICANT counts
    siggies = []

    for i in range(0, var_count):
        
        # Compute ALEXANDER-GOVERN Test
        AG_result = alexandergovern(df.iloc[:,i], df.iloc[:,-1])
        # Use getattr to select p-value 
        p_value = getattr(AG_result,'pvalue')
                       
        # TEMPORARY: Save p-value significance into list
        if p_value < 0.05:
            siggies.append(True)
        else:
            siggies.append(False)
    return siggies

# Testing consistency of ALEXANDER-GOVERN test

AG_sigs = []

for i in range(16, 25, 1):
    # Loop through Poisson-distributed categorical features in stacked_all
    y_train = df.iloc[:168, i]
    
    # Run ALEXANDER-GOVERN TEST
    AG_sigs.append({
        "Poisson Label": i,
        "Significants List": alexandergovern_fvl(X_train, y_train)})
    
from pprint import pprint
pprint(AG_sigs)

# Extract and flatten the 'Significants List' values
AG_flat_list = [value for item in AG_sigs for value in item['Significants List']]

# Count the number of True and False values
AG_num_true = sum(AG_flat_list)
AG_num_false = len(AG_flat_list) - AG_num_true

print("Number of True values:", AG_num_true)
print("Number of False values:", AG_num_false)

################## Graphing ####################################

# Data
categories = ['ANOVA', 'Kruskal-Wallis H Test', 'Alexander-Govern']
sigs = [ANOVA_num_true, KRUSKAL_num_true, AG_num_true]
no_sigs = [ANOVA_num_false, KRUSKAL_num_false, AG_num_false]

# Number of categories
n = len(categories)

# X axis locations for the groups
ind = np.arange(n)

# Width of the bars
width = 0.35

# Plotting
fig, ax = plt.subplots()

# Bars for Significant
bar1 = ax.bar(ind - width/2, sigs, width, label='Significant', color='skyblue')

# Bars for Not Significant
bar2 = ax.bar(ind + width/2, no_sigs, width, label='Not Significant', color='salmon')

# Adding labels, title, and legend
ax.set_xlabel('Tests')
ax.set_ylabel('Significance Counts')
# ax.set_title('ANOVA v. KWH Test v. AG Test\n on GAUSSIAN Columns & POISSON Label')
ax.set_xticks(ind)
ax.set_xticklabels(categories)
ax.legend()

# Save plot
plt.savefig('C:/Users/aceme/OneDrive/Documents/GitHub/BP24/AceMejiaSanchez/Images/ANOVA_Alts_Gaussian.png', dpi=300)

# Show plot
plt.show() 

################## KRUSKAL-WALLIS H Test: Further Investigation #######################################

### Creating Scatterplot

# # Assigning variables to graph: KWH's NOT SIGNIFICANT
# x = df.iloc[:,0] # Gaussian var
# y = df.iloc[:, 1] # Gaussian var
# z = df.iloc[:, 23] # Poisson var

# Assigning variables to graph: KWH's SIGNIFICANT
x = df.iloc[:,0] # Gaussian var
y = df.iloc[:, 1] # Gaussian var
z = df.iloc[:, 24] # Poisson var

plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y, c=z, cmap='viridis', edgecolors="black", alpha=0.75)
#plt.colorbar(scatter, label='Intensity')  # Add colorbar indicating intensity

plt.title('Scatter plot comparing GAUSSIAN Columns v. POISSON LABEL')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

# Save plot
#plt.savefig('C:/Users/aceme/OneDrive/Documents/GitHub/BP24/AceMejiaSanchez/Images/scatterplot_GAUSSIAN_v_Poisson_KWH_NOT_SIGNIFICANT.png', dpi=300)
plt.savefig('C:/Users/aceme/OneDrive/Documents/GitHub/BP24/AceMejiaSanchez/Images/scatterplot_GAUSSIAN_v_Poisson_KWH_SIGNIFICANT.png', dpi=300)


plt.show()

# ################## KRUSKAL-WALLIS H Test (FvF) #######################################

# print("\n------------------ Kruskal-Wallis H Test (Feature vs Feature) -----------------------")
# # Determines if all features in X_train have same mean via ranks
# def KWH_fvf(X_train):
        
#     # Extract variable names
#     variable_names = list(X_train.columns)

#     # Initialize matrices to store chi-squared and p-values
#     num_variables = len(variable_names)
#     kwh_stats = np.zeros((num_variables, num_variables))
#     p_values = np.zeros((num_variables, num_variables))

#     # Compute chi-squared and p-values for each pair of variables
#     for i, j in combinations(range(num_variables), 2):
        
#         # Compute KRUSKA-WALLIS H Test
#         kwh, p = kruskal(X_train.iloc[:, i], X_train.iloc[:, j])
        
#         # Assign results to kwh_stats and p_values matrices
#         kwh_stats[i, j] = kwh
#         kwh_stats[j, i] = kwh  # Assign to symmetric position in the matrix
#         p_values[i, j] = p
#         p_values[j, i] = p  # Assign to symmetric position in the matrix

#     # Create a DataFrame with variable names as index and columns
#     kwh_stats_df = pd.DataFrame(kwh_stats, index=variable_names, columns=variable_names)
#     p_values_df = pd.DataFrame(p_values, index=variable_names, columns=variable_names)

#     # Printing the matrix-like output with variable names
#     print("\nF-Statistics:")
#     print(kwh_stats_df)
#     print("\nP-Values:")
#     print(p_values_df)
    