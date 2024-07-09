# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:07:08 2024

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
################## Import function ######################################################
import sys
sys.path.append('C:/Users/aceme/OneDrive/Documents/GitHub/BP24/')
import Measure_Patterns

# ################## Import data ######################################################
# df = pd.read_csv("C:/Users/aceme/OneDrive/Documents/SIAM Simons Summer Opportunity/Datasets/WVS_Cross-National_Wave_7_csv_v6_0.csv")

# ################## Data cleaning ######################################################
# # Find index of Q1 to last column & label (Q46)
# start_col_index = df.columns.get_loc('Q1')
# label_index = df.columns.get_loc('Q46')

# # Subsetting to only columns that are Core Questions to Contextual Questinos in WVS CodeBook
# df = df.iloc[:,start_col_index:-1]

# # Finding columns that are strings
# string_vars = df.select_dtypes(include=['object'])
# string_vars.columns

# # Dropping columns that are strings/objects, these seem to be country-specific variables
# df = df.drop(columns=['X002_02B', 'V002A_01', 'V001A_01', 'Partyname', 'Partyabb', 'CPARTY',
#        'CPARTYABB'])

# # Find index of Q1-Q290 & label (Q46)
# start_col_index = df.columns.get_loc('Q1')
# label_index = df.columns.get_loc('Q46')

# # Checking classes of Q1 & Q46
# old_q1 = df.Q1.unique()
# old_q46 = df.Q46.unique()

# # Define the mapping for missing values
# value_mapping = {
#     -1: np.nan,
#     -2: np.nan,
#     -4: np.nan,
#     -5: np.nan,
#     -999.0: np.nan,
#     -9999.0: np.nan
# }

# # Function to apply the mapping
# def map_binary_values(x):
#     return value_mapping.get(x, x)

# # Apply the mapping only to the specified columns
# df.iloc[:,start_col_index:-1] = df.iloc[:,start_col_index:-1].applymap(map_binary_values)

# # Define the mapping for missing values
# Q46_value_mapping = {
#      1: 1,
#      2: 1,
#      3: 0,
#      4: 0,
# }

# # # Function to apply the mapping
# def map_Q46_values(x):
#     return Q46_value_mapping.get(x, x)

# # # Apply the mapping only to the specified column Q46
# df.iloc[:,label_index] = df.iloc[:,label_index].map(map_Q46_values)

# # Re-checking classes of Q1 & Q46
# new_q1 = df.Q1.unique()
# new_q46 = df.Q46.unique()

# print("Old Classes: \n", old_q1, old_q46)
# print("New Classes: \n", new_q1, new_q46)

# # Drop rows where 'Q46' has NaN values
# df = df.dropna(subset=['Q46'])

# # Checking if it works
# print(df['Q46'].isna().sum())

# # Checking which columns have 80,000+ NaNs
# # Calculate the sum of NaNs in each column
# # nan_counts = df.isna().sum()

# # Print the result for each column
# #for column, count in nan_counts.items():
#     #print(f'Column "{column}" has {count} NaN values.')

# # Keeping original shape
# old_shape = df.shape

# # Define the ranges of column names to drop
# ranges_to_drop = [
#     ('Q82_AFRICANUNION', 'Q82_UNDP'),  # First range
#     ('Q291G1', 'Q294B'),  # Second range
#     ('ID_GPS', 'v2xnp_client')   # Third range
# ]

# # Create a list to store column names to drop
# columns_to_drop = []

# # Populate the list with column names from the defined ranges
# for start_col, end_col in ranges_to_drop:
#     start_idx = df.columns.get_loc(start_col)
#     end_idx = df.columns.get_loc(end_col)
#     columns_to_drop.extend(df.columns[start_idx:end_idx+1])

# # Drop the columns
# df = df.drop(columns=columns_to_drop)

# # Dropping all rows that contain NAs
# df = df.dropna()
# new_shape = df.shape
# print("Old Shape:", old_shape)
# print("New Shape:", new_shape)

# # Check for columns without negative values
# non_negative_columns = df.columns[(df >= 0).all()]

# # Subset the DataFrame to include only columns with negative values
# df = df[non_negative_columns]

# df_shape = df.shape
# print("Old Shape:", old_shape)
# print("New Shape:", new_shape)

################## Split dataset into X_train and y_train ####################################
# X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,start_col_index:-1].drop(columns=['Q46']), df.iloc[:,label_index], test_size=0.2, random_state=42)

################## Running Measure_Patterns() ####################################
# Measure_Patterns.MeasurePatterns(X_train, y_train)

################## ANOVA FvL Re-Write ####################################

# Import data
df = pd.read_csv("C:/Users/aceme/OneDrive/Documents/GitHub/BP24/Ellee/Sanity Checks/Demos/stacked_all.csv")

# Indexing through pre-prepared splitting in stacked_all
X_train = df.iloc[:168, :9]
# X_test = df.iloc[168:241, :9]
# y_train =  df.iloc[:168, 22]
# y_test = df.iloc[168:241, 22]

# print("\n------------------ ANOVA (Feature vs Label) -----------------------")

# # Finds dependency between all features in X_train & the label in y_train
# def anova_fvl(X_train, y_train):
    
#     # Combining X_train and y_train
#     df = X_train
#     df['y_train'] = y_train

#     # Number of features, excluding label
#     var_count = len(X_train.columns)-1

#     # Creates an empty array for f-statistic and P-values
#     results = []

#     for i in range(0, var_count):
        
#         # Compute ANOVA
#         f_statistic, p_value = f_oneway(df.iloc[:,i], df.iloc[:,-1])
        
#         # Save p-value significance into list
#         if p_value < 0.05:
#             significance = "Significant"
#         else:
#             significance = "Not Significant"
           
#         # Append results to the list
#         results.append({
#             "Feature": df.columns[i],
#             "F-Statistic": f_statistic,
#             "P-Value": p_value, 
#             "Significance": significance})

#     # Create a dataFrame from the results
#     results_df = pd.DataFrame(results)
    
#     # Print the dataFrame
#     print("Label:", df.columns.values[-1])
#     print(results_df.to_string(index=False))

# # Testing consistency of ANOVA test
# for i in range(16, 25, 1):
#     # Loop through Poisson-distributed categorical features in stacked_all
#     y_train = df.iloc[:168, i]
    
#     # Run ANOVA
#     anova_fvl(X_train, y_train)
    
    
################## ANOVA FvF Re-Write ####################################

print("\n------------------ANOVA for Features v. Features-----------------------")
# Finds dependency between all features in X_train
def anova_fvf(X_train):
        
    # Extract variable names
    variable_names = list(X_train.columns)

    # Initialize matrices to store chi-squared and p-values
    num_variables = len(variable_names)
    f_stats = np.zeros((num_variables, num_variables))
    p_values = np.zeros((num_variables, num_variables))

    # Compute chi-squared and p-values for each pair of variables
    for i, j in combinations(range(num_variables), 2):

        # Compute ANOVA: f-statistics and p-values
        f, p = f_oneway(X_train.iloc[:, i], X_train.iloc[:, j])
            
        # Assign results to f_stats and p_values matrices
        f_stats[i, j] = f
        f_stats[j, i] = f  # Assign to symmetric position in the matrix
        p_values[i, j] = p
        p_values[j, i] = p  # Assign to symmetric position in the matrix

    # Create a DataFrame with variable names as index and columns
    f_stats_df = pd.DataFrame(f_stats, index=variable_names, columns=variable_names)
    p_values_df = pd.DataFrame(p_values, index=variable_names, columns=variable_names)

    # Printing the matrix-like output with variable names
    print("\nF-Statistics:")
    print(f_stats_df)
    print("\nP-Values:")
    print(p_values_df)
    