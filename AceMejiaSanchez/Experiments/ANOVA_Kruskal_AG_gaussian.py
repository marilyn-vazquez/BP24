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
import Measure_Patterns

################## Import Data & X_train ######################################

# Import data
df = pd.read_csv("C:/Users/aceme/OneDrive/Documents/GitHub/BP24/Ellee/Sanity Checks/Demos/stacked_all.csv")

# Indexing through pre-prepared splitting in stacked_all
X_train = df.iloc[:168, :8]
# X_test = df.iloc[168:241, :8]
# y_train =  df.iloc[:168, 22]
# y_test = df.iloc[168:241, 22]

################## ANOVA FvL Re-Write #########################################

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
        
        # # Save p-value significance into list
        # if p_value < 0.05:
        #     significance = "Significant"
        # else:
        #     significance = "Not Significant"
           
        # # Append results to the list
        # results.append({
        #     "Feature": df.columns[i],
        #     "F-Statistic": f_statistic,
        #     "P-Value": p_value, 
        #     "Significance": significance})

    # # Create a dataFrame from the results
    # results_df = pd.DataFrame(results)
    
    # # Print the dataFrame
    # print("Label:", df.columns.values[-1])
    # print(results_df.to_string(index=False))

# Testing consistency of ANOVA test

# ANOVA_sigs = []

# for i in range(16, 25, 1):
#     # Loop through Poisson-distributed categorical features in stacked_all
#     y_train = df.iloc[:168, i]
    
#     # Run ANOVA
#     ANOVA_sigs.append(anova_fvl(X_train, y_train))
    
# print(ANOVA_sigs)

# # Flatten the list of lists
# ANOVA_flat_list = [item for sublist in ANOVA_sigs for item in sublist]

# # Count the number of True and False values
# ANOVA_num_true = sum(ANOVA_flat_list)
# ANOVA_num_false = len(ANOVA_flat_list) - ANOVA_num_true

# print("Number of True values:", ANOVA_num_true)
# print("Number of False values:", ANOVA_num_false)


# ################## KRUSKAL-WALLIS H Test (FvL) #######################################

# print("\n------------------ Kruskal-Wallis H Test (Feature vs Label) -----------------------")

# # Finds dependency between all features in X_train & the label in y_train
# def kruskal_fvl(X_train, y_train):
    
#     # Combining X_train and y_train
#     df = X_train
#     df['y_train'] = y_train

#     # Number of features, excluding label
#     var_count = len(X_train.columns)-1
    
#     # TEMPORARY: Creates an empty array for tracking SIGNIFICANT counts
#     siggies = []

#     for i in range(0, var_count):
        
#         # Compute KRUSKA-WALLIS H Test
#         kruskal_statistic, p_value = kruskal(df.iloc[:,i], df.iloc[:,-1])
                                
#         # TEMPORARY: Save p-value significance into list
#         if p_value < 0.05:
#             siggies.append(True)
#         else:
#             siggies.append(False)
#     return siggies

# # Testing consistency of ANOVA test

# KRUSKAL_sigs = []

# for i in range(16, 25, 1):
#     # Loop through Poisson-distributed categorical features in stacked_all
#     y_train = df.iloc[:168, i]
    
#     # Run ANOVA
#     KRUSKAL_sigs.append(kruskal_fvl(X_train, y_train))
    
# print(KRUSKAL_sigs)

# # Flatten the list of lists
# KRUSKAL_flat_list = [item for sublist in KRUSKAL_sigs for item in sublist]

# # Count the number of True and False values
# KRUSKAL_num_true = sum(KRUSKAL_flat_list)
# KRUSKAL_num_false = len(KRUSKAL_flat_list) - KRUSKAL_num_true

# print("Number of True values:", KRUSKAL_num_true)
# print("Number of False values:", KRUSKAL_num_false)

# ################## ALEXANDER-GOVERN Test #######################################

# print("\n------------------ ALEXANDER-GOVERN Test (Feature vs Label) -----------------------")

# # Finds dependency between all features in X_train & the label in y_train
# def alexandergovern_fvl(X_train, y_train):
    
#     # Combining X_train and y_train
#     df = X_train
#     df['y_train'] = y_train

#     # Number of features, excluding label
#     var_count = len(X_train.columns)-1
    
#     # TEMPORARY: Creates an empty array for tracking SIGNIFICANT counts
#     siggies = []

#     for i in range(0, var_count):
        
#         # Compute ALEXANDER-GOVERN Test
#         AG_result = alexandergovern(df.iloc[:,i], df.iloc[:,-1])
#         # Use getattr to select p-value 
#         p_value = getattr(AG_result,'pvalue')
                       
#         # TEMPORARY: Save p-value significance into list
#         if p_value < 0.05:
#             siggies.append(True)
#         else:
#             siggies.append(False)
#     return siggies

# # Testing consistency of ALEXANDER-GOVERN test

# AG_sigs = []

# for i in range(16, 25, 1):
#     # Loop through Poisson-distributed categorical features in stacked_all
#     y_train = df.iloc[:168, i]
    
#     # Run ALEXANDER-GOVERN TEST
#     AG_sigs.append(alexandergovern_fvl(X_train, y_train))
    
# print(AG_sigs)

# # Flatten the list of lists
# AG_flat_list = [item for sublist in AG_sigs for item in sublist]

# # Count the number of True and False values
# AG_num_true = sum(AG_flat_list)
# AG_num_false = len(AG_flat_list) - AG_num_true

# print("Number of True values:", AG_num_true)
# print("Number of False values:", AG_num_false)

# ################## Graphing ####################################

# # Data
# categories = ['ANOVA', 'Kruskal-Wallis H Test', 'Anderson-Govern']
# sigs = [ANOVA_num_true, KRUSKAL_num_true, AG_num_true]
# no_sigs = [ANOVA_num_false, KRUSKAL_num_false, AG_num_false]

# # Number of categories
# n = len(categories)

# # X axis locations for the groups
# ind = np.arange(n)

# # Width of the bars
# width = 0.35

# # Plotting
# fig, ax = plt.subplots()

# # Bars for Significant
# bar1 = ax.bar(ind - width/2, sigs, width, label='Significant')

# # Bars for Not Significant
# bar2 = ax.bar(ind + width/2, no_sigs, width, label='Not Significant')

# # Adding labels, title, and legend
# ax.set_xlabel('Tests')
# ax.set_ylabel('Counts')
# ax.set_title('GAUSSIAN: ANOVA v. Kruskal-Wallis v. Anderson-Govern')
# ax.set_xticks(ind)
# ax.set_xticklabels(categories)
# ax.legend()

# # Show plot
# plt.show()

################## KRUSKAL-WALLIS H Test: Further Investigation #######################################

### Creating Scatterplot

# Assigning variables to graph
x = df.iloc[:,13] # Uniform var
y = df.iloc[:,15] # Uniform var
z = df.iloc[:, 24] # Poisson var
# kruskal(df.iloc[:,1], df.iloc[:,22])

plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y, c=z, cmap='viridis', alpha=0.75)
plt.colorbar(scatter, label='Intensity')  # Add colorbar indicating intensity

plt.title('Scatter plot with color based on a third variable')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()

################## KRUSKAL-WALLIS H Test (FvF) #######################################

print("\n------------------ Kruskal-Wallis H Test (Feature vs Feature) -----------------------")
# Determines if all features in X_train have same mean via ranks
def KWH_fvf(X_train):
        
    # Extract variable names
    variable_names = list(X_train.columns)

    # Initialize matrices to store chi-squared and p-values
    num_variables = len(variable_names)
    kwh_stats = np.zeros((num_variables, num_variables))
    p_values = np.zeros((num_variables, num_variables))

    # Compute chi-squared and p-values for each pair of variables
    for i, j in combinations(range(num_variables), 2):
        
        # Compute KRUSKA-WALLIS H Test
        kwh, p = kruskal(X_train.iloc[:, i], X_train.iloc[:, j])
        
        # Assign results to kwh_stats and p_values matrices
        kwh_stats[i, j] = kwh
        kwh_stats[j, i] = kwh  # Assign to symmetric position in the matrix
        p_values[i, j] = p
        p_values[j, i] = p  # Assign to symmetric position in the matrix

    # Create a DataFrame with variable names as index and columns
    kwh_stats_df = pd.DataFrame(kwh_stats, index=variable_names, columns=variable_names)
    p_values_df = pd.DataFrame(p_values, index=variable_names, columns=variable_names)

    # Printing the matrix-like output with variable names
    print("\nF-Statistics:")
    print(kwh_stats_df)
    print("\nP-Values:")
    print(p_values_df)
    