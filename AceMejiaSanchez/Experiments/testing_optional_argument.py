# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 14:54:38 2024

@author: aceme
"""
import sys
sys.path.append('C:/Users/aceme/OneDrive/Documents/GitHub/BP24/')
import Fake_Patterns
from sklearn import datasets
import pandas as pd
import random

iris = datasets.load_iris()

# Convert the iris dataset to a pandas dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add the target variable to the dataframe
df['target'] = iris.target

# Print the first 5 rows of the dataframe
print(df.head())


############################ Testing optional argument ####################################
optional_test = []

for i in range(5):
    optional_test.append(bool(random.getrandbits(1)))

# Call the measure_patterns function
Fake_Patterns.FakePatterns(X_train=df.iloc[0:4], y_train=df['target'], optional=optional_test)