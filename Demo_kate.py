################## Import function ############################################
import sys
import pandas as pd
import numpy as np
sys.path.append("C:/Users/kateh/OneDrive/Documents/GitHub/BP24/Fake_Differences.py")
import Fake_Differences

################## Import data ################################################
df1 = pd.read_csv("C:/Users/kateh/OneDrive/Documents/GitHub/BP24/Ellee/Data/Gaussian/gaussian_orig.csv", header=None)
df_aug = pd.read_csv("C:/Users/kateh/OneDrive/Documents/GitHub/BP24/AugSynDatasets/gaussian_randswap.csv", header=None)
df_test = pd.read_csv("C:/Users/kateh/OneDrive/Documents/GitHub/BP24/AugSynDatasets/gaussian_test.csv", header=None)
array1 = df_aug.to_numpy()
array2 = df_test.to_numpy()
array = np.vstack((array1, array2))
df2 = pd.DataFrame(array)
################## Data cleaning ##############################################
# Whatever changes you make to your data set

################# Categorical and Numerical Columns ###########################
# Whatever subsetting you make to your columns
columns_to_convert = [2, 3, 7, 9, 12]

for col in columns_to_convert:
    df1.iloc[:, col] = df1.iloc[:, col].astype('category')
    df2.iloc[:, col] = df2.iloc[:, col].astype('category')

################## Running FakeDifferences() #################################
Fake_Differences.FakeDifferences(df1, df2)