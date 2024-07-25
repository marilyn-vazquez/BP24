# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:47:10 2024

@author: aceme
"""

################## Import relevant packages ###################################
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

################## Import function ############################################
import sys
sys.path.append('C:/Users/aceme/OneDrive/Documents/GitHub/BP24/')
import Fake_Measures

################## Import data ################################################
df = pd.read_csv("C:/Users/aceme/OneDrive/Documents/GitHub/BP24/DEMO_WVS_data.csv")

################# Split dataset into X_train and y_train ######################
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], 
                                                    df.iloc[:,-1], 
                                                    test_size=0.2, 
                                                    random_state=42)

################# Running Measure_Patterns() ##################################
Fake_Measures.FakeMeasures(features=X_train, label=y_train)