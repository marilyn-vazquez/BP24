# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 15:44:37 2022

@author: jeffr
"""

import numpy as np
import pandas as pd
import random


'''
DESCRIPTION: applies chosen augmentation method to supplied dataframe.
It is assumed that that the labels column is located at the END of the dataframe.

INPUTS:
data = dataframe
nrows = number of rows that will be created in augmentation

OPTIONAL:
nvalues = number of values being swapped, limited to column length

PMONE and GAUSNOISE ONLY:
unit = amount being plus or minused
noise = 
'''

def betterApplyAugmentationMethods(data, method, nrows, nvalues=None, unit=None, noise=None):
    
    # If nvalues not specified, entire column is selected
    if nvalues == None:
        nvalues = data.shape[1]-1
    
    if str(method).lower() == 'pmone':
        # Creates empty dataframe to store augmented data
        augmentedDf = pd.DataFrame()
        
        # Randomly selects rows from data and appends to augmentedDf
        for i in range(nrows):
            augmentedDf = pd.concat([augmentedDf, data.iloc[[random.randint(0, data.shape[0]-1)]]], ignore_index=True)
            
        # Drops labels column in augmentedDf
        augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
        
        # Selects nvalues amount of unique column indexes
        randCols = random.sample(range(0, data.shape[1]-1), nvalues)
        
        # Iterates through augmentedData and applies plus or minus to randCols indexes
        for i in range(augmentedDf.shape[0]):
            for col in randCols:
                if (random.randint(0, 1) == 0):
                    augmentedDf.iloc[i, col] += unit
                else:
                    augmentedDf.iloc[i, col] -= unit

        # Combines data and augmentedDf into finished augmented dataframe
        augmentedDf = pd.concat([data, augmentedDf], ignore_index=True)
        
        return augmentedDf
    
    elif str(method).lower() == 'modpmone':
        # Creates empty dataframe to store augmented data
        augmentedDf = pd.DataFrame()
        
        # Randomly selects rows from data and appends to augmentedDf
        for i in range(nrows):
            augmentedDf = pd.concat([augmentedDf, data.iloc[[random.randint(0, data.shape[0]-1)]]], ignore_index=True)
            
        # Drops labels column in augmentedDf
        augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
        
        # Selects nvalues amount of unique column indexes
        randCols = random.sample(range(0, data.shape[1]-1), nvalues)
        
        # Iterates through augmentedData and applies plus or minus to randCols indexes
        for i in range(augmentedDf.shape[0]):
            for col in randCols:
                colMax = data.iloc[:, col].max()
                colMin = data.iloc[:, col].min()
                
                if (augmentedDf.iloc[i, col] + unit < colMax and augmentedDf.iloc[i, col] - unit > colMin):
                    if (random.randint(0, 1) == 0):
                        if (augmentedDf.iloc[i, col] + unit <= colMax):
                            augmentedDf.iloc[i, col] += unit
                        else:
                            augmentedDf.iloc[i, col] -= unit
                    else:
                        if (augmentedDf.iloc[i, col] - unit >= colMin):
                            augmentedDf.iloc[i, col] -= unit
                        else:
                            augmentedDf.iloc[i, col] += unit

        # Combines data and augmentedDf into finished augmented dataframe
        augmentedDf = pd.concat([data, augmentedDf], ignore_index=True)
        
        return augmentedDf
    
    elif str(method).lower() == 'gausnoise':
        # Creates empty dataframe to hold augmented rows
        augmentedDf = pd.DataFrame()
        
        # Selects random rows from data and appends to augmentedDf
        for i in range(nrows):
            augmentedDf = pd.concat([augmentedDf, data.iloc[[random.randint(0, data.shape[0]-1)]]], ignore_index=True)
        
        # Drops label column of augmentedDf
        augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
        
        # Selects random unique column index 
        randCols = random.sample(range(0, data.shape[1]-1), nvalues)
        
        # Applies Gaussian noise to randCols values stored in array
        for i in range(augmentedDf.shape[0]):
            for cols in randCols:
                augmentedDf.iloc[i, cols] += np.random.normal(0, noise)
            
        
        # Combines both data and augmentedDf for full augmented dataframe
        augmentedDf = pd.concat([data, augmentedDf], ignore_index=True)
            
        
        return augmentedDf
    
    elif str(method).lower() == 'modgausnoise':
        # Creates an empty dataframe to hold augmented observations
        augmentedDf = pd.DataFrame()
        
        # Randomly selects unique column indexs from data
        randCols = random.sample(range(0, data.shape[1]-1), nvalues)
        
        # Appends randomly selected rows from data to augmentedDf
        for i in range(nrows):
            augmentedDf = pd.concat([augmentedDf, data.iloc[[random.randint(0, data.shape[0]-1)]]], ignore_index=True)
            
        # Drops labels from augmentedDf
        augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
        
        # Generates Gaussian distribution based on columns summary statistics
        # Swaps value with random value in generated Gaussian distribution
        for col in randCols:
            for i in range(augmentedDf.shape[0]):
                mean = augmentedDf[col].mean()
                stDev = augmentedDf[col].std()
                
                augmentedDf.iloc[i, col] =  np.random.normal(mean, stDev)
            
        # Combines both data and augmentedDf into final dataframe
        augmentedDf = pd.concat([data, augmentedDf], axis=0, ignore_index=True)
        
        return augmentedDf
    
    elif str(method).lower() == 'randswap':
        # Creates empty dataframe to store augmented rows
        augmentedDf = pd.DataFrame()
        
        # Copies nrows from original data and appends to augmentedDf
        for i in range(nrows):
            augmentedDf = pd.concat([augmentedDf, data.iloc[[random.randint(0, data.shape[0]-1)]]], ignore_index=True)
            
        # Drops labels column from augmentedDF
        augmentedDf = augmentedDf.drop(augmentedDf.shape[1]-1, axis=1)
        
        # Picks UNIQUE column indexes to swap
        columnIndexSwaps = random.sample(range(0, data.shape[1]-1), nvalues)

        # Swaps augmentedDf column value from same column in data
        for i in range(augmentedDf.shape[0]):
            for col in columnIndexSwaps:
                randValue = data.iloc[random.randint(0, data.shape[0]-1), col]
                
                augmentedDf.iloc[i, col] = randValue
            
        # Combines both data and augmentedDf into one dataframe
        augmentedDf = pd.concat([data, augmentedDf], axis=0, ignore_index=True)
        
        return augmentedDf
    
    else:
        print("Method not found")
        return None