#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:32:57 2024

@author: fabianafazio
"""
# upload libraries
import numpy as np
import pandas as pd
from numpy import linalg
from itertools import combinations
from scipy import stats


# STEPS TO DO IN A     SEPARATE     FILE:
    
################## Import function ############################################
import sys
sys.path.append('/Users/fabianafazio/Documents/GitHub/BP24/')
import Fake_Differences

################## Import data ################################################
dataset1 = pd.read_csv("/Users/fabianafazio/Documents/GitHub/BP24/Ellee/Data/Gaussian/gaussian_orig.csv", header=None)
dataset2 = pd.read_csv("/Users/fabianafazio/Documents/GitHub/BP24/Ellee/Data/Gaussian/gaussian_new.csv", header=None)

################# Categorical and Numerical Columns ###########################

# DATA 1 (SYNTHETIC)
# Whatever subsetting you make to your columns
# Define the primary columns to move to the end
primary_indices = [2, 3, 7, 9, 12]
primary_columns = dataset1.columns[primary_indices]
# Define the initial columns, excluding the primary columns
initial_columns = [col for col in dataset1.columns if col not in primary_columns]
# Combine the initial columns with the primary columns
new_column_order = initial_columns + list(primary_columns)
# Reorder the DataFrame
dataset1 = dataset1[new_column_order]
# Grab the last 13 columns
data1 = dataset1.iloc[:, -13:]
# Reset the column index to go from 0 to 12
data1.columns = range(13)
# Display the first few rows of the new DataFrame
print(data1.dtypes)



# DATA 2 (AUGMENTED)
# Define the primary columns to move to the end
primary_indices = [2, 3, 7, 9, 12]
primary_columns = dataset2.columns[primary_indices]
# Define the initial columns, excluding the primary columns
initial_columns = [col for col in dataset2.columns if col not in primary_columns]
# Combine the initial columns with the primary columns
new_column_order = initial_columns + list(primary_columns)
# Reorder the DataFrame
dataset2 = dataset2[new_column_order]
# Grab the last 13 columns
data2 = dataset2.iloc[:, -13:]
# Apply rounding and conversion to integers for the last 5 columns
data2.iloc[:, -5:] = dataset2.iloc[:, -5:].round().astype(int)  # only for new generated points
# Reset the column index to go from 0 to 12
data2 = range(13)
# Display the first few rows of the new DataFrame
print(data2)


################## Data cleaning ##############################################
# Whatever changes you make to your data set

# Convert the last 5 columns to integers
#for col in data1.columns[-5:]:
#    data1[col] = data1[col].astype(int)
#print(data1.dtypes)
# Splitting numerical subset 
#numerical_df = data1.select_dtypes(include = ['float', 'float64'])
# Splitting categorical subset 
#categorical_df = data1.select_dtypes(exclude=['float', 'float64'])


# Convert the last 5 columns to integers
#for col in data2.columns[-5:]:
#    data2[col] = data2[col].astype(int)
#print(data2.dtypes)
# Splitting numerical subset 
#numerical_df = data2.select_dtypes(include = ['float', 'float64'])
# Splitting categorical subset 
#categorical_df = data2.select_dtypes(exclude=['float', 'float64'])



################## Running FakeDifferences() #################################
Fake_Differences.FakeDifferences(data1, data2)


#################### DO NOT UNCOMMENT!!!!!!!! #################################