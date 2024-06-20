
import pdb
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from scipy import stats
from itertools import combinations
from sklearn.datasets import load_iris

"""
The function Measure_Patterns has 3 parameters: X_train, y_train, optional
optional will check if the columns selected is categorical (integers and strings) or numerical (float)
if optional is not provided, then the program will assume that the column has integers values, therefore it will be considered categorical
"""


# Load dataset 
data = np.loadtxt("C:/Users/aceme/OneDrive/Documents/GitHub/BP24/Data Creation/Gaussian - small distance/gaussian_small_d_1.tex")
# Creating NumPy array
array = np.array(data)
# Converting to Pandas DataFrame
df_table = pd.DataFrame(array)
# Displaying the table
print(df_table)



# From the dataset, change 25 columns to 'categorical'
#Loop, converts floats to ints and then those ints to category
for i in range(26):
    df_table.iloc[:,i] = df_table.iloc[:,i].astype(int)
    df_table.iloc[:,i] = df_table.iloc[:,i].astype("category")
df_table.iloc[:, 150] = df_table.iloc[:, 150].astype("category")




# Split dataset into X_train and y_train
#X_train, X_test, y_train, y_test = train_test_split(df_table.iloc[:,1:150], df_table.iloc[:,-1], test_size=0.2, random_state=52)


# Function Measure_Patterns begins here!
def Measure_Patterns(X_train, y_train, optional=None):
    
    # Check if the data type is provided for columns
    if optional is None:
        print("Optional parameter not provided. Assuming integers values are categorical")
    
        # Splitting X_train into numerical subset 
        numerical_df = X_train.select_dtypes(include = ["float64"])

        # Splitting X_train into categorical subset 
        categorical_df = X_train.select_dtypes(exclude=['float64'])
    else:
        # Create empty numerical & categorical data frames
        numerical = []
        numerical_colnames = []
        categorical = []
        categorical_colnames = []
        
        # Check that length of optional = # of columns in X_train
        # Optional is the column type for X_train, so the lengths should be equal
        if len(optional) == len(X_train.columns):
            # For all the values in optional
            for i in range(len(optional)):
                if optional[i] == True:
                    numerical.append(X_train.iloc[:,i])
                    # TO DO: Save SPECIFIC column name in each loop, order matters
                    # numerical_colnames.append()
                else: 
                    categorical.append(X_train.iloc[:,i])
                    # TO DO: Save SPECIFIC column name in each loop, order matters
                    # categorical_colnames.append()
            # Turn transposed arrays into dataframes
            numerical_df = pd.DataFrame(np.transpose(numerical))
            categorical_df = pd.DataFrame(np.transpose(categorical))
            # TO DO: Re-attach the column names to numerical_df & categorical_df 
           
            print("Numerical DF:")
            print(numerical_df)
            print("Categorical Df")
            print(categorical_df)
            
        else:
            print("The length of X_train and optional are different.")
            
            
#####################################################################

from sklearn import datasets
import pandas as pd

# load iris dataset
iris = datasets.load_iris()
# Since this is a bunch, create a dataframe
iris_df=pd.DataFrame(iris.data)
iris_df['class']=iris.target

iris_df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
iris_df.dropna(how="all", inplace=True) # remove any empty lines

# categoris v nums
optional_test = [True, True, True, False]


# Split dataset into X_train and y_train
X_train, X_test, y_train, y_test = train_test_split(iris_df.iloc[:,0:4], iris_df.iloc[:,-1], test_size=0.2, random_state=52)

# Call the measure_patterns function
Measure_Patterns(X_train, y_train, optional=optional_test)
