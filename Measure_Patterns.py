# -*- coding: utf-8 -*-

import pdb
import random
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from scipy import stats
from itertools import combinations
from sklearn.datasets import load_iris
from scipy.stats import anderson

"""
The function Measure_Patterns has 3 parameters: X_train, y_train, optional
optional will check if the columns selected is categorical (integers and strings) or numerical (float)
if optional is not provided, then the program will assume that the column has integers values, therefore it will be considered categorical
"""


# Load dataset 
data = np.loadtxt("/Users/fabianafazio/Documents/GitHub/BP24/Fabiana/Demos Fabi/uniform_large_d_1.tex")
# Creating NumPy array
array = np.array(data)
# Converting to Pandas DataFrame
df_table = pd.DataFrame(array)
# Displaying the table
print(df_table)



# From the dataset, change 25 columns to 'categorical'
#Loop, converts floats to ints and then those ints to category
for i in range(25):
    df_table.iloc[:,i] = df_table.iloc[:,i].round()
    df_table.iloc[:,i] = df_table.iloc[:,i].astype(int)
    df_table.iloc[:,i] = df_table.iloc[:,i].astype("category")
df_table.iloc[:, 150] = df_table.iloc[:, 150].astype("category")




# Split dataset into X_train and y_train
X_train, X_test, y_train, y_test = train_test_split(df_table.iloc[:,0:150], df_table.iloc[:,-1], test_size=0.2, random_state=52)


# Function Measure_Patterns begins here!
def Measure_Patterns(X_train, y_train, optional=None):
    
    # Check if the data type is provided for columns
    if optional is None:
        print("Optional parameter not provided. Assuming integers values are categorical")
    
        # Splitting X_train into numerical subset 
        print("\nNumerical DataFrame:")
        numerical_df = X_train.select_dtypes(include = ["float64"])
        print(numerical_df)

        # Splitting X_train into categorical subset 
        print("Categorical DataFrame:")
        categorical_df = X_train.select_dtypes(exclude=['float64'])
        print(categorical_df)
    

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
                    # Save numerical column in numerical list
                    numerical.append(X_train.iloc[:,i])
                    # Save numerical column name
                    numerical_colnames.append(X_train.columns[i]) 
                else: 
                    # Save categorical column in numerical list
                    categorical.append(X_train.iloc[:,i])
                    # Save categorical column name
                    categorical_colnames.append(X_train.columns[i])
            # Turn transposed arrays into dataframes
            numerical_df = pd.DataFrame(np.transpose(numerical))
            categorical_df = pd.DataFrame(np.transpose(categorical))
            # Re-attach the column names to numerical_df & categorical_df 
            numerical_df.columns = numerical_colnames
            categorical_df.columns = categorical_colnames

            print("Numerical DF:")
            print(numerical_df)
            print("Categorical Df")
            print(categorical_df)
            
        else:
            print("The length of X_train and optional are different.")
            
     

##################### Correlation between columns (numerical) Code ############################
    # Takes the X_train data to find correlation between NUMERICAL features
    def num_corr(X_train_numerical):
        matrix = X_train_numerical.corr(method='pearson')
        print("---------------------------Correlation Matrix------------------------- \n", matrix)
     
    #Calls the function so the matrix prints out    
    num_corr(numerical_df)
    
##################### Chi-Square (F vs F) Code ################################################
    
    print("\n------------------Chi-Squared for Features v. Features-----------------------")
    # Finds dependency between all features in X_train
    def chi_squared_fvf(X_train):
            
        # Extract variable names
        variable_names = list(X_train.columns)
    
        # Initialize matrices to store chi-squared and p-values
        num_variables = len(variable_names)
        chi_squared = np.zeros((num_variables, num_variables))
        p_values = np.zeros((num_variables, num_variables))
    
        # Creates an empty boolean array for contingency table cells <5
        below_5 = []
    
        # Compute chi-squared and p-values for each pair of variables
        for i, j in combinations(range(num_variables), 2):
            
            # Creates contigency table of feature i v. feature j
            contingency_table = pd.crosstab(X_train.iloc[:, i], X_train.iloc[:, j])
            
            # Check if any cell in the contingency table is below 5 & appends
            below_5.append((contingency_table < 5).any().any())
            
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
    
        # Print warning if any cells are below 5
        if any(below_5):
            print("WARNING: The validity of this chi-squared test may be violated as there are \n         cells below 5 in at least one contingency table of observed values.") 
        
        # Printing the matrix-like output with variable names
        print("Chi-Squared Values:")
        print(chi_squared_df)
        print("\nP-Values:")
        print(p_values_df)
        
    chi_squared_fvf(categorical_df)
    
##################### Chi-Square (F vs label column) Code ####################################
    
    print("\n------------------------Chi-Square (F vs label column)------------------------")
    # Finds dependency between all features in X_train & the label in y_train
    def chi_squared_fvl(X_train, y_train):
            
        # Combining X_train and y_train
        df = X_train
        df['label'] = y_train
    
        # Number of features, excluding label
        var_count = len(df.columns)-1
    
        # Creates an empty array for Chi2 and P-values
        results = []
    
        # Creates an empty boolean array for contingency table cells <5
        below_5 = []
    
        for i in range(0, var_count):
    
            # Create contigency table of all features v. label
            contingency_table = pd.crosstab(df.iloc[:, i], df.iloc[:,-1])
            
            # Check if any cell in the contingency table is below 5 & appends
            below_5.append((contingency_table < 5).any().any())
                
            # Compute chi-squared and p-values
            chi2 = stats.chi2_contingency(contingency_table)[0]
            p = stats.chi2_contingency(contingency_table)[1]
                
            # Append results to the list
            results.append({
                "Feature": df.columns[i],
                "Chi Squared Statistic": chi2,
                "P-Value": p})
    
        # Create a dataFrame from the results
        results_df = pd.DataFrame(results)
    
        # Print warning if any cells are below 5
        if any(below_5):
            print("WARNING: The validity of this chi-squared test may be violated as there are \n         cells below 5 in your contingency table of observed values.") 
        
        # Print the dataFrame
        print("Label:", df.columns.values[-1])
        print(results_df.to_string(index=False))
        
    chi_squared_fvl(categorical_df, y_train)
    
    
    ############################# Anderson-Darling Test ##########################
    # tests if a sample comes from a population with a specific distribution
    # used to determine whether or not your data follow a normal distribution
    from scipy.stats import anderson

    # Subset to select only numerical variables columns --> A-D Test only works with numerical
    df_DA = X_train.select_dtypes(include = ["float64"])

    # Get the actual column indices for the numerical columns
    numerical_column_indices = [X_train.columns.get_loc(col) for col in df_DA.columns]

    # Initialize a list to store results
    results = []

    # Significance level for the normality test (usually 0.05)
    significance_level_index = 2  # Index for 5% significance level in the Anderson-Darling test

    # Iterate through each row
    for col_index, column in zip(numerical_column_indices, df_DA.columns):
        # Convert columns to a numpy array
        data = df_DA[column].values

        # Perform the Anderson-Darling Test
        result = anderson(data)

        # Determine if the distribution is normal at the 5% significance level
        is_normal = result.statistic < result.critical_values[significance_level_index]

        #H0:  the data are normally distributed, 
        #Ha:  the data are not normally distributed. 
        # Formulate the hypothesis result
        hypothesis = "H0: Fail to reject" if is_normal else "Ha: Reject"
    
        # Store the results
        results.append({
            'feature': col_index,
            'statistic': result.statistic,
            'critical_values': result.critical_values,
            'significance_level': result.significance_level,
            'normal_dist': is_normal,
            'hypothesis': hypothesis
            })

    # Convert results to a DataFrame for better readability
    results_df = pd.DataFrame(results)

    # Display the results
    print("---------------------------------------- Anderson-Darling Test Results ---------------------------------------------")
    print(results_df.to_string())
    
    
########################## Histogram/Graphing ###############################

print("------------------------Histogram/Graphing-----------------------------")


###### These are just here for now so the histogram and bar graph functions work
# Splitting X_train into numerical subset 
numerical_df = X_train.select_dtypes(include = ["float64"])

# Splitting X_train into categorical subset 
categorical_df = X_train.select_dtypes(exclude=['float64'])
        

# Ensure data is 2D
if numerical_df.ndim == 1:
    numerical_df = numerical_df.reshape(-1, 1)  # Reshape 1D array to 2D array with one column

# Number of features (columns) in the dataset
numerical_num_features = numerical_df.shape[1]


# Loop through each numerical feature
for feature_idx in range(numerical_num_features):
    # Extract the current feature data (column)
    feature_df = numerical_df.iloc[:, feature_idx]

    # Compute histogram with 10 bins
    hist, bin_edges = np.histogram(feature_df, bins=10)

    # Print feature number
    print(f"Feature {feature_idx + 1}:")
    
    # Print bin edges
    print("Bin Edges:", bin_edges)

    # Store bin heights in a list
    bin_heights = []
    bin_heights.extend(hist)
    print("Array with bin heights:", bin_heights)

    # Store bin probabilities in a list and normalize
    bin_probs = []
    bin_probs.extend(hist)
    bin_probs = np.array(bin_probs) / sum(bin_heights)
    print("Array with bin probabilities:", bin_probs)

    # Loop through each bin to print range and probabilities
    for i in range(len(hist)):
        bin_range = f"{bin_edges[i]:.2f} to {bin_edges[i+1]:.2f}"  # Bin range
        bin_probability = hist[i] / sum(hist)  # Bin probability
        print(f"Bin {i + 1} ({bin_range}): Height = {hist[i]}, Probability = {bin_probability:.2f}")

    # Separator between features for clarity
    print("\n" + "="*50 + "\n")

# Calculate and store probabilities for each categorical column
print("Proportions for Label for Categorical Columns:")

for column in categorical_df.columns:
    value_counts = categorical_df[column].value_counts(normalize=True).sort_index()
    # print(f"Probabilities for Categorical Column {column}:")
    print(value_counts)
    print()  # Add an empty line for separation    

############################ KL Divergence ####################################

optional_test = []

for i in range(151):
    optional_test.append(bool(random.getrandbits(1)))

# Call the measure_patterns function
Measure_Patterns(X_train, y_train)
