################## Import function ############################################
import sys
import pandas as pd
sys.path.append("C:/Users/kateh/OneDrive/Documents/GitHub/BP24/Fake_Patterns.py")
import Fake_Differences

################## Import data ################################################
df1 = pd.read_csv("C:/Users/kateh/OneDrive/Documents/GitHub/BP24/Ellee/Data/Gaussian/gaussian_orig.csv", header=None)
df2 = pd.read_csv("C:/Users/kateh/OneDrive/Documents/GitHub/BP24/AugSynDatasets/gaussian_randswap.csv", header=None)

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