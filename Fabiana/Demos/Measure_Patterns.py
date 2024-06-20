# -*- coding: utf-8 -*-

import pdb
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
data = np.loadtxt("uniform_large_d_1.tex")
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
X_train, X_test, y_train, y_test = train_test_split(df_table.iloc[:,0:150], df_table.iloc[:,-1], test_size=0.2, random_state=52)


#true if nums
#false otherwise
#[true false true]
#num = data.iloc['']
#cat = data.iloc['']


# Function Measure_Patterns begins here!
def Measure_Patterns(X_train, y_train, optional=None):
    
    # Check if the data type is provided for columns
    if optional is None:
        print("Optional parameter not provided. Assuming integers values are categorical")
        optional = {}
    
        # Splitting X_train into numerical subset 
        print("\nNumerical DataFrame:")
        numerical_df = X_train.select_dtypes(include = ["float64"])
        print(numerical_df)

        # Splitting X_train into categorical subset 
        print("Categorical DataFrame:")
        categorical_df = X_train.select_dtypes(exclude=['float64'])
        print(categorical_df)
    


##################### Correlation between columns (numerical) Code ############################
    # Takes the X_train data to find correlation between NUMERICAL features
    def num_corr(X_train_numerical):
        matrix = X_train_numerical.corr(method='pearson')
        print("---------------------------Correlation Matrix------------------------- \n", matrix)
     
    #Calls the function so the matrix prints out    
    num_corr(numerical_df)
    
##################### Chi-Square (F vs F) Code ################################################
    
    print("\n------------------Chi-Squared for Features v. Features-----------------------")
    # Finds dependency between all CATEGORICAL features in X_train
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
    
    chi_squared_fvf(categorical_df)
    
##################### Chi-Square (F vs label column) Code ####################################
    
    print("\n------------------------Chi-Square (F vs label column)------------------------")
    # Finds dependency between all CATEGORICAL features in X_train & the label in y_train
    def chi_squared_fvl(X_train_categorical, y_train):
        
        # Combining CATEGORICAL X_train and y_train
        df = X_train_categorical
        df['label'] = y_train

        # Number of features, excluding label
        var_count = len(df.columns)-1

        # Creates an empty array to print values in a table
        results = []

        for i in range(0, var_count):

            # Create contigency table of all features v. label
            crosstab = pd.crosstab(df.iloc[:, i], df.iloc[:,-1])
            
            # Compute chi-squared and p-values
            chi2 = stats.chi2_contingency(crosstab)[0]
            p = stats.chi2_contingency(crosstab)[1]
            
            # Append results to the list
            results.append({
                "Feature": df.columns[i],
                "Chi Squared Statistic": chi2,
                "P-Value": p})

        # Create a DataFrame from the results
        results_df = pd.DataFrame(results)

        # Print the DataFrame
        print("Label:", df.columns.values[-1])
        print(results_df.to_string(index=False))
    
    chi_squared_fvl(categorical_df, y_train)
################################# KS Test ###########################################
    print("\n---------------------Kolmogorov Smirnov Test--------------------------")

    # Subset to select only numerical variables columns --> KS Test only works with numerical
    # Subset to select only numerical variables columns --> KS Test only works with numerical
    df_KS = df_table.select_dtypes(include = ["float64"])
    # Add label column to new KS dataset to compare Feature to Label
    label_column = df_table.iloc[:, -1]
    df_KS['label_column'] = label_column
    df_KS.head()

    def standardize(sample):
        return (sample - np.mean(sample)) / np.std(sample)

    def ks_test(sample):
        # Sort the sample
        sample_sorted = np.sort(sample)
        # Evaluate the empirical CDF (ECDF)
        ecdf = np.arange(1, len(sample_sorted)+1) / len(sample_sorted)
        # Evaluate the theoretical CDF
        cdf = stats.norm.cdf(sample_sorted)
        # Calculate the KS statistic
        ks_stat = np.max(np.abs(ecdf - cdf))
        # Calculate the p-value
        p_value = stats.kstest(sample_sorted, 'norm').pvalue
        return ks_stat, p_value
    
    # Temporary KS Test
    var_count = len(df_KS.columns)-1

    #creates an empty array to print values in a table
    results = [] 

    for i in range(0, var_count):
        # Select one feature from the dataset (Example: assuming the first column is numeric)
        sample = df_KS.iloc[:, i]  # Change the column index as needed
        # Standardize the sample
        standardized_sample = standardize(sample)
        # Perform the KS test on standardize sample
        ks_stat, p_value = ks_test(standardized_sample)
        # Determine if we reject or fail to reject the null hypothesis
        # If sample does not come from a normal distribution ---> reject H0
        # If sample comes from a normal distribution ---> fail to reject H0
        normal_dist = p_value > 0.05
        hypothesis_result = "Fail to reject H0" if normal_dist else "Reject H0"

        # Append results to the list
        results.append({
            "Feature": df_KS.columns[i],
            "KS Statistic": f"{ks_stat:.4f}",
            "P-Value": f"{p_value:.3e}",
            "Normal Distribution": normal_dist,
            "Hypothesis Result": hypothesis_result})
    
    # Create a DataFrame from the results
    results_df = pd.DataFrame(results)

    # Print the DataFrame
    print(results_df.to_string(index=False))
    
########################## Histogram/Graphing ###############################

print("------------------------Histogram/Graphing-----------------------------")
# Ensure data is 2D
if data.ndim == 1:
    data = data.reshape(-1, 1)  # Reshape 1D array to 2D array with one column

# Number of features (columns) in the dataset
num_features = data.shape[1]

# Loop through each feature
for feature_idx in range(num_features):
    # Extract the current feature data (column)
    feature_data = data[:, feature_idx]

    # Compute histogram with 10 bins
    hist, bin_edges = np.histogram(feature_data, bins=10)

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

############################ KL Divergence ####################################












# Call the measure_patterns function
Measure_Patterns(X_train, y_train)

