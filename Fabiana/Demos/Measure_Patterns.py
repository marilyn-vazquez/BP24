# -*- coding: utf-8 -*-

import pdb
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split


"""
The function Measure_Patterns has 3 parameters: X_train, y_train, optional
optional will check if the columns selected is categorical (integers and strings) or numerical (float)
if optional is not provided, then the program will assume that the column has integers values, therefore it will be considered categorical
"""

#load dataset 
data = np.loadtxt("uniform_large_d_1.tex")
# Creating NumPy array
array = np.array(data)
# Converting to Pandas DataFrame
df_table = pd.DataFrame(array)
# Displaying the table
print(df_table)




# Split dataset into X_train and y_train
X_train, X_test, y_train, y_test = train_test_split(df_table.iloc[:,1:150], df_table.iloc[:,-1], test_size=0.2, random_state=52)




# Function Measure_Patterns begins here!
def Measure_Patterns(X_train, y_train, optional=None):
    # Check if the data type is provided for columns
    if optional is None:
        print("Optional parameter not provided. Assuming integers values are categorical")
        optional = {}
    # Classify columns based on their data type
    def classify_columns(column):
        if np.issubdtype(column.dtype, np.number):
            if column.dtype == "float":
                return "numerical"
            else:
                return "categorical"
        else:
            return "categorical"
    
    # Default factory function which returns 'categorical' for any key not found in 'optional'
    column_types = defaultdict(lambda: "categorical", {col: classify_columns(X_train[col]) for col in X_train.columns})

    # Update column_types with any specific types from the optional dictionary
    column_types.update(optional)
    
    # Create a list to store column information
    columns_info = [{'Column': col, 'Type': column_types[col]} for col in X_train.columns]
    
    # Create a DataFrame from the columns information
    columns_info_df = pd.DataFrame(columns_info)
    
    # Print the DataFrame
    print(columns_info_df)



    # Correlation between columns (numerical + categorical) Code
    
    
    # Chi-Square (F vs F / F vs label column) Code
    
    
    #KS Test
    
    
    #KL Diverge












# Call the measure_patterns function
Measure_Patterns(X_train, y_train)