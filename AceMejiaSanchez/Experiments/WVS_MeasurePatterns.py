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
from scipy.stats import kruskal
################## Import function ######################################################
import sys
sys.path.append('C:/Users/aceme/OneDrive/Documents/GitHub/BP24/')
import Fake_Measures

################## Import data ######################################################
df = pd.read_csv("C:/Users/aceme/OneDrive/Documents/SIAM Simons Summer Opportunity/Datasets/WVS_Cross-National_Wave_7_csv_v6_0.csv")

################## Data cleaning ######################################################
# Find index of Q1 to last column & label (Q46)
start_col_index = df.columns.get_loc('Q1')
label_index = df.columns.get_loc('Q46')

# Subsetting to only columns that are Core Questions to Contextual Questinos in WVS CodeBook
df = df.iloc[:,start_col_index:-1]

# Finding columns that are strings
string_vars = df.select_dtypes(include=['object'])
string_vars.columns

# Dropping columns that are strings/objects, these seem to be country-specific variables
df = df.drop(columns=['X002_02B', 'V002A_01', 'V001A_01', 'Partyname', 'Partyabb', 'CPARTY',
        'CPARTYABB'])

# Find index of Q1-Q290 & label (Q46)
start_col_index = df.columns.get_loc('Q1')
label_index = df.columns.get_loc('Q46')

# Checking classes of Q1 & Q46
old_q1 = df.Q1.unique()
old_q46 = df.Q46.unique()

########### Keep track of column types bc mapping with np.nan will reduce to all floats
# Identify integer columns and store the list of columns
int_columns = [col for col in df.columns if pd.api.types.is_integer_dtype(df[col])]
# Some columns are eliminated later in the code, so they are being removed from this list
cols_to_remove = ['Q82_AFRICANUNION', 'Q82_APEC', 'Q82_ARABLEAGUE', 'Q82_ASEAN', 'Q82_CIS', 'Q82_CUSMA', 'Q82_ECO', 'Q82_EU', 'Q82_GULFCOOP', 'Q82_ISLCOOP', 'Q82_MERCOSUR', 'Q82_NAFTA', 'Q82_OAS', 'Q82_SAARC', 'Q82_SCO', 'Q82_TLC', 'Q82_UNDP', 'Q276', 'Q276A', 'Q276R', 'Q280', 'Q282', 'Q284', 'Q291G1', 'Q291G2', 'Q291G3', 'Q291G4', 'Q291G5', 'Q291G6', 'Q291P1', 'Q291P2', 'Q291P3', 'Q291P4', 'Q291P5', 'Q291P6', 'Q291UN1', 'Q291UN2', 'Q291UN3', 'Q291UN4', 'Q291UN5', 'Q291UN6', 'Q292A', 'Q292B', 'Q292C', 'Q292D', 'Q292E', 'Q292F', 'Q292G', 'Q292H', 'Q292I', 'Q292J', 'Q292K', 'Q292L', 'Q292M', 'Q292N', 'Q292O', 'Q293', 'Q294A', 'Q294B']
integer_columns = [item for item in int_columns if item not in cols_to_remove]

# Identify object columns and store the list of columns
object_columns = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]

# Define the mapping for missing values
value_mapping = {
    -1: np.nan,
    -2: np.nan,
    -4: np.nan,
    -5: np.nan,
    -999.0: np.nan,
    -9999.0: np.nan
}

# Function to apply the mapping
def map_binary_values(x):
    return value_mapping.get(x, x)

# Apply the mapping only to the specified columns
df.iloc[:,start_col_index:-1] = df.iloc[:,start_col_index:-1].applymap(map_binary_values)

# Define the mapping for missing values
Q46_value_mapping = {
      1: 1,
      2: 1,
      3: 0,
      4: 0,
}

# # Function to apply the mapping
def map_Q46_values(x):
    return Q46_value_mapping.get(x, x)

# # Apply the mapping only to the specified column Q46
df.iloc[:,label_index] = df.iloc[:,label_index].map(map_Q46_values)

# Re-checking classes of Q1 & Q46
new_q1 = df.Q1.unique()
new_q46 = df.Q46.unique()

print("Old Classes: \n", old_q1, old_q46)
print("New Classes: \n", new_q1, new_q46)

# Drop rows where 'Q46' has NaN values
df = df.dropna(subset=['Q46'])

# Checking if it works
print(df['Q46'].isna().sum())

# Checking which columns have 80,000+ NaNs
# Calculate the sum of NaNs in each column
# nan_counts = df.isna().sum()

# Print the result for each column
#for column, count in nan_counts.items():
    #print(f'Column "{column}" has {count} NaN values.')

# Keeping original shape
old_shape = df.shape

# Define the ranges of column names to drop
ranges_to_drop = [
    ('Q82_AFRICANUNION', 'Q82_UNDP'),  # First range
    ('Q291G1', 'Q294B'),  # Second range
    ('ID_GPS', 'v2xnp_client')   # Third range
]

# Create a list to store column names to drop
columns_to_drop = []

# Populate the list with column names from the defined ranges
for start_col, end_col in ranges_to_drop:
    start_idx = df.columns.get_loc(start_col)
    end_idx = df.columns.get_loc(end_col)
    columns_to_drop.extend(df.columns[start_idx:end_idx+1])

# Drop the columns
df = df.drop(columns=columns_to_drop)

# Dropping all rows that contain NAs
df = df.dropna()
new_shape = df.shape
print("Old Shape:", old_shape)
print("New Shape:", new_shape)

# Check for columns without negative values
non_negative_columns = df.columns[(df >= 0).all()]

# Subset the DataFrame to include only columns with negative values
df = df[non_negative_columns]

df_shape = df.shape
print("Old Shape:", old_shape)
print("New Shape:", new_shape)

### Convert the stored list of columns back to integer & object types
df[integer_columns] = df[integer_columns].astype(int)
df[object_columns] = df[object_columns].astype('object')

# Saving clean data as csv file
# df.to_csv('C:/Users/aceme/OneDrive/Documents/GitHub/BP24/DEMO_WVS_data.csv', index=False)

# ################# Split dataset into X_train and y_train ####################################
# X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,start_col_index:-1].drop(columns=['Q46']), df.iloc[:,label_index], test_size=0.2, random_state=42)

# ################# Running Measure_Patterns() ####################################
# Fake_Patterns.FakePatterns(X_train, y_train)


