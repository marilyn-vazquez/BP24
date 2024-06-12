# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 14:39:19 2022

@author: jeffr
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

# Makes sure the random numbers are the same each time the 
# program runs
np.random.seed(1)

def generateRawData(nrows, ncolumns, distance, distribution):
    if distribution == "uniform":
        
        # The first points will always be between 0 and 1
        a = 0
        b = 1
        
        # Second set of points is a either 0 or 1, the distance between points,
        # and 1 added together
        c = a + distance 
        d = b + distance 

        # Create two separate dataframes that fit in two different 
        # ranges on a uniform distribution
        df1 = pd.DataFrame(np.random.uniform(a, b, (int(nrows / 2), ncolumns)))
        df1['labels'] = 0
        df2 = pd.DataFrame(np.random.uniform(c, d, (int(nrows/2), ncolumns)))
        df2['labels'] = 1
        
        # Concatenate the data frames and mix the rows together
        df = pd.concat([df1, df2])
        perm = np.random.permutation(df)
        
        # Turn the permutated data back into a dataframe for use
        dfreal = pd.DataFrame(perm)
        
        # Save the dataframe to a text file if others want to use
        # np.savetxt('synthetic_data_with_labels.txt', dfreal)
        
        # fig, ax = plt.subplots()
        # plt.scatter(df[0], df[1])
        # plt.title("Random Uniform Distribution")
        # plt.show()
        
        return dfreal
    else:
        # Creates first Gaussian distribution
        label1 = pd.DataFrame(np.random.normal(5 + distance, 1, size=(int(nrows/2), ncolumns)))
        # Adds new column for label
        label1['label'] = 0
        
        
        # Creates second Gaussian distribution
        label2 = pd.DataFrame(np.random.normal(5, 1, size=(int(nrows/2), ncolumns)))
        # Adds new column for label
        label2['label'] = 1
        
        # Combines both Gaussian distributions
        df = pd.concat([label1, label2])
        
        
        # Shuffles Gaussian distributions
        shuffled_df = pd.DataFrame(np.random.permutation(df))
            
        # Creates historgram of Gaussian distributions
        # plt.scatter(shuffled_df[0], shuffled_df[1])
        # plt.title("Random Gaussian Distribution")
        # plt.show()
        
        # Saves generated Gaussian distribution in same folder as file
        # np.savetxt("Generated Gaussian Distribution.txt", shuffled_df)
        
        return shuffled_df

# raw_uniform = generateRawData(500, 150, 2, "uniform")
# raw_gaussian = generateRawData(500, 150, 2, "gaussian")