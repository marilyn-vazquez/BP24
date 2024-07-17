#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 20:32:57 2024

The function FakeDifferences does the following :
    
  Parameters:
    - data1 is the sythetic dataset (before augmentation)
    - data2 is the augmented dataset
    - optional is a boolean list where TRUE indicates a CATEGORICAL variable.
    This will check if the columns selected are categorical (integers and strings) 
    or numerical (floats). If optional is not provided, then the program will assume that 
    the column has integers values, therefore it will be considered categorical
    
    
  What does this function do?:
    - This function takes 2 datasets and compares the differences between them
    - Correlation between columns test is for (Feature vs Feature) and tests the 
    Frobenius norm which computes the difference between the correlation matrices
    - Chi-square test is for (Feature vs Feature) which compares only the feature columns in the dataset 
    that are categorical (excluding the label column). 
    - Chi-square test is for (Feature vs Label) which compares the categorical columns to 
    the label column
    
  Output:
    - Correlation between columns test outputs 2 matrices correlations that only takes
    the numerical columns in the dataset. This outputs the differences between both correlations
    by printing the absolute error and relative error betweem them
    - Chi-square test (Feature vs Feature) outputs the Chi-square significance values in a dataframe, 
    and the p-value in another dataframe. Both take only the catgorical columns in the dataset. These
    dataframes change to a True and False dataframe which only uses the p-values. 
    The final output is the count of changes in the True and False table.
       - If the p-value is less than 0.05 there is no relationship (This is the H0, prints TRUE)
       - If the p-value is greater than 0.05 there us a relationship (This is the Ha, prints FALSE)
    - Chi-square test (Feature vs Label) outputs the the Chi-square significance values and p-values
    in 2 columns. Both take only the catgorical columns in the dataset. The test only grabs the p-values
    and changes them to True and False. 
    The final output is the count of changes in the True and False table.
       - If the p-value is less than 0.05 there is no relationship (This is the H0, prints TRUE)
       - If the p-value is greater than 0.05 there us a relationship (This is the Ha, prints FALSE)
    


"""
# upload libraries
import numpy as np
import pandas as pd
from numpy import linalg
from itertools import combinations
from scipy import stats


    
####################### Import function FakeDifferences ##############################
import sys
sys.path.append('/Users/fabianafazio/Documents/GitHub/BP24/Fake_Patterns.py')
import Fake_Differences


############################### Import data ##########################################
data1 = pd.read_csv("/Users/fabianafazio/Documents/GitHub/BP24/Ellee/Data/Gaussian/gaussian_orig.csv", header=None)
data2 = pd.read_csv("/Users/fabianafazio/Documents/GitHub/BP24/AugSynDatasets/gaussian_randswap.csv", header=None)


################# Categorical and Numerical Columns ###########################

columns_to_convert = [2, 3, 7, 9, 12]

for col in columns_to_convert:
    data1.iloc[:, col] = data1.iloc[:, col].astype('category')
    data2.iloc[:, col] = data2.iloc[:, col].astype('category')


################## Data cleaning ##############################################
# Whatever changes you make to your data set



################## Running FakeDifferences() #################################
Fake_Differences.FakeDifferences(data1, data2)


