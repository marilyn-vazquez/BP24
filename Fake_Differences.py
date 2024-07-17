#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:55:06 2024

FakeDifferences measures the following:
    - The differences in matrices with the Correlation Matrix Test (Feature vs Feature)
    - The count of changes in p-values with the Chi-Square (Feature vs Feature)
    - The count of changes in p-values with the Chi-Square (Feature vs Label)


@author: fabianafazio
"""

# Upload libraries
import numpy as np
import pandas as pd
from numpy import linalg
from itertools import combinations
from scipy import stats

# STEPS TO DO IN A     SEPARATE     FILE:
    
################## Import function ############################################
# import sys
# sys.path.append('/Users/...YOUR FILE PATH HERE FOR FAKE_DIFFERENCES.../')
# import Fake_Differences

################## Import data ################################################
# df = pd.read_csv("C:/Users/...YOUR dataset FILE PATH HERE")

################## Data cleaning ##############################################
# Whatever changes you make to your data set

################# Categorical and Numerical Columns ###########################
# Whatever subsetting you make to your columns

################## Running FakeDifferences() #################################
# Fake_Differences.FakeDifferences(data1, data2)

#################### DO NOT UNCOMMENT!!!!!!!! #################################



def FakeDifferences(data1, data2, optional=None):
    # Initialize empty dataframes for numerical and categorical data for both datasets
    numerical_df1 = pd.DataFrame()
    categorical_df1 = pd.DataFrame()
    numerical_df2 = pd.DataFrame()
    categorical_df2 = pd.DataFrame()
    
    # Function to split data into numerical and categorical dataframes
    def split_data(data, optional):
        numerical_df = pd.DataFrame()
        categorical_df = pd.DataFrame()
        
        if optional is None:
            print("Optional parameter not provided. Assuming integer values are categorical")
            numerical_df = data.select_dtypes(include=['float', 'float64'])
            categorical_df = data.select_dtypes(exclude=['float', 'float64'])
        else:
            numerical = []
            numerical_colnames = []
            categorical = []
            categorical_colnames = []

            if len(optional) == len(data.columns):
                for i in range(len(optional)):
                    if optional[i] == True:
                        numerical.append(data.iloc[:, i])
                        numerical_colnames.append(data.columns[i])
                    else:
                        categorical.append(data.iloc[:, i])
                        categorical_colnames.append(data.columns[i])
                numerical_df = pd.DataFrame(np.transpose(numerical))
                categorical_df = pd.DataFrame(np.transpose(categorical))
                numerical_df.columns = numerical_colnames
                categorical_df.columns = categorical_colnames
            else:
                print("The length of data and optional are different.")
        
        return numerical_df, categorical_df

    # Split both datasets
    numerical_df1, categorical_df1 = split_data(data1, optional)
    numerical_df2, categorical_df2 = split_data(data2, optional)
    
    # Print numerical and categorical dataframes for both datasets
    print("\nNumerical DataFrame 1:")
    print(numerical_df1)
    print("\nCategorical DataFrame 1:")
    print(categorical_df1)
    
    print("\nNumerical DataFrame 2:")
    print(numerical_df2)
    print("\nCategorical DataFrame 2:")
    print(categorical_df2)






    ############################## Correlation between columns (Feature vs Feature)  ##################################
    # Function to find correlation between numerical features
    def num_corr(X_train_numerical):
        matrix = X_train_numerical.corr(method='pearson')
        print("---------------------------Correlation Matrix------------------------- \n", matrix)
        return matrix

    # Correlation matrix for numerical data1
    correlation_matrix1 = num_corr(numerical_df1)
    correlation_df1 = pd.DataFrame(correlation_matrix1)
    print("\nCorrelation DataFrame 1:")
    print(correlation_df1)
    print(f"Shape: {correlation_df1.shape}")
    
    # Correlation matrix for numerical data2
    correlation_matrix2 = num_corr(numerical_df2)
    correlation_df2 = pd.DataFrame(correlation_matrix2)
    print("\nCorrelation DataFrame 2:")
    print(correlation_df2)
    print(f"Shape: {correlation_df2.shape}")
    
    # Convert the dataframes to numpy arrays
    matrix1 = correlation_df1.to_numpy()
    matrix2 = correlation_df2.to_numpy()

    # Compute the Frobenius norm of the difference between the matrices
    frobenius_abs = np.linalg.norm(matrix1 - matrix2, ord='fro')   # Absolute error with Frobenius norm
    frobenius_rel = frobenius_abs / np.linalg.norm(matrix1, ord='fro')    # Relative error with Frobenius norm

    print(f"Frobenius norm (absolute error): {frobenius_abs:.3f}")
    print(f"Frobenius norm (relative error): {frobenius_rel:.3f}")





    ################################# Chi-Square (Feature vs Feature)  ##########################################
    print("\n------------------Chi-Squared for Features v. Features-----------------------")
    # Function to find dependency between all categorical features
    def chi_squared_fvf(X_train_categorical):
        # Extract variable names
        variable_names = list(X_train_categorical.columns)

        # Initialize matrices to store chi-squared and p-values
        num_variables = len(variable_names)
        chi_squared = np.zeros((num_variables, num_variables))
        p_values = np.zeros((num_variables, num_variables))

        # Compute chi-squared and p-values for each pair of variables
        for i, j in combinations(range(num_variables), 2):
            contingency_table = pd.crosstab(X_train_categorical.iloc[:, i], X_train_categorical.iloc[:, j])
            
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

        return p_values_df

    # Chi-Square test for categorical data
    p_values_df1 = chi_squared_fvf(categorical_df1)
    p_values_df2 = chi_squared_fvf(categorical_df2)

    # Create a new DataFrame with True/False based on the p_value condition
    print("----------- Chi-Square (F vs F) True and False for Data1 ------------")
    p_value_df1 = p_values_df1 < 0.05
    print(p_value_df1)

    print("----------- Chi-Square (F vs F) True and False for Data2 ------------")
    p_value_df2 = p_values_df2 < 0.05
    print(p_value_df2)

    # Count the changes between the two DataFrames
    changes = (p_value_df1 != p_value_df2).sum().sum()

    # Display the number of changes
    print(f"Number of changes between data1 and data2 in Chi-Square (F vs F): {changes}")






    
    ################################### Chi-Square (Feature vs Label)  ##########################################
    # Extract y_train (label column)
    y_train1 = data1.iloc[:, 12]
    y_train2 = data2.iloc[:, 12]
    
    print("\n------------------------Chi-Square (F vs label column)------------------------")
    # Function to find dependency between all categorical features and the label
    def chi_squared_fvl(X_train_categorical, y_train):
        # Combining categorical X_train and y_train
        df = X_train_categorical.copy()
        df['label'] = y_train

        # Number of features, excluding label
        var_count = len(df.columns) - 1

        # Creates an empty array to print values in a table
        results = []

        for i in range(var_count):
            # Create contingency table of all features v. label
            crosstab = pd.crosstab(df.iloc[:, i], df.iloc[:, -1])
            
            # Compute chi-squared and p-values
            chi2 = stats.chi2_contingency(crosstab)[0]
            p = stats.chi2_contingency(crosstab)[1]
            
            # Append results to the list
            results.append({
                "Feature": df.columns[i],
                "Chi Squared Statistic": chi2,
                "P-Value": p
            })

        # Create a DataFrame from the results
        results_df = pd.DataFrame(results)

        # Print the DataFrame
        print("Label:", df.columns.values[-1])
        print(results_df.to_string(index=False))

        return results_df
    
    results_df1 = chi_squared_fvl(categorical_df1, y_train1)
    results_df2 = chi_squared_fvl(categorical_df2, y_train2)

    print("----------- Chi-Square (F vs L) True and False for Data1 ------------")
    p_value_fvl_df1 = results_df1['P-Value'] < 0.05
    print(p_value_fvl_df1)

    print("----------- Chi-Square (F vs L) True and False for Data2 ------------")
    p_value_fvl_df2 = results_df2['P-Value'] < 0.05
    print(p_value_fvl_df2)
    
    # Count the changes between the two DataFrames
    changes = (p_value_fvl_df1 != p_value_fvl_df2).sum()

    # Display the number of changes
    print(f"Number of changes between data1 and data2 in Chi-Square (F vs L): {changes}")
    
    
    
    ###########################################################################


