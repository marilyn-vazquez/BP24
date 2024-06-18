# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:00:15 2024

@author: aceme
"""

import os
import pandas as pd
import numpy as np
import scipy.stats as stats

data = np.loadtxt("C:/Users/aceme/OneDrive/Documents/GitHub/BP24/Data Creation/Gaussian - small distance/gaussian_small_d_1.tex")

# Creating NumPy array
array = np.array(data)

# Converting to Pandas DataFrame
df = pd.DataFrame(array)

# Look at data
df.head()

############## Creating small data set #############################

# Subsetting to last 15 columns that includes the label
df = df.iloc[:, 135:151]

# Converting the first 5 columns from floats -> integers -> categories
for i in range(0, 5):

    df.iloc[:,i] = df.iloc[:,i].astype(int) # Integer
    df.iloc[:,i] = df.iloc[:,i].astype('category') # Categories
    
# Turn label into categorical label
df.iloc[:,15] = df.iloc[:,15].astype('category')

# Creating subset of only CATEGORICAL variables + LABEL
df_categorical = df.select_dtypes(include=['category'])
df_categorical['label'] = df.iloc[:,15]
df_categorical

############### CHI-SQUARE TEST FOR ALL FEATURES V. ALL FEATURES ##############

from itertools import combinations

data = df_categorical

# Extract variable names
variable_names = list(data.columns)

# Initialize matrices to store chi-squared and p-values
num_variables = len(variable_names)
chi_squared = np.zeros((num_variables, num_variables))
p_values = np.zeros((num_variables, num_variables))

# Compute chi-squared and p-values for each pair of variables
for i, j in combinations(range(num_variables), 2):
    contingency_table = pd.crosstab(data.iloc[:, i], data.iloc[:, j])
    
    # Compute chi-squared and p-values
    chi2 = stats.chi2_contingency(contingency_table)[0]
    p = stats.chi2_contingency(contingency_table)[1]
    
    # Assign results to chi_squared and p_values matrices
    chi_squared[i, j] = chi2
    chi_squared[j, i] = chi2  # Assign to symmetric position in the matrix
    p_values[i, j] = p
    p_values[j, i] = p  # Assign to symmetric position in the matrix

# Create a DataFrame with variable names as index and columns
chi_squared_df = pd.DataFrame(chi_squared, index=variable_names, columns=variable_names)
p_values_df = pd.DataFrame(p_values, index=variable_names, columns=variable_names)

# Printing the matrix-like output with variable names
print("Chi-Squared Values:")
print(chi_squared_df)
print("\nP-Values:")
print(p_values_df)

############### CHI-SQUARE TEST FOR LABEL V. ALL FEATURES #####################


# Number of features, excluding label
var_count = len(df_categorical.columns)-1


# Creates an empty array to print values in a table
results = []

for i in range(0, var_count):

    # Create contigency table of all features v. label
    crosstab = pd.crosstab(df_categorical.iloc[:, i], df_categorical.iloc[:,-1])
    
    # Compute chi-squared and p-values
    chi2 = stats.chi2_contingency(crosstab)[0]
    p = stats.chi2_contingency(crosstab)[1]
    
    # Append results to the list
    results.append({
        "Feature": df_categorical.columns[i],
        "Chi Squared Statistic": chi2,
        "P-Value": p})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Print the DataFrame
print("Label:", df_categorical.columns.values[-1])
print(results_df.to_string(index=False))