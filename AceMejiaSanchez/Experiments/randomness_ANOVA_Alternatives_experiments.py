# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:02:17 2024

@author: aceme
"""

################## Import relevant packages####################################
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import random
from scipy.stats import f_oneway
from scipy.stats import friedmanchisquare
from itertools import combinations
from scipy.stats import kruskal
from scipy.stats import alexandergovern
from scipy.stats import ttest_ind
from random import sample 

################## Set Seed ####################################################
# Set a seed for reproducibility
np.random.seed(42)
    
################## Data #########################################

# Binomial column
n = 10
prob = .5
binomial_500 = np.random.binomial(n, prob, 500)

# Poisson column
lamda = 1
poisson_500 = np.random.poisson(lamda, 500)

# Gaussian column
gaussian_1 = np.random.normal(100, 50, 250)
gaussian_2 = np.random.normal(13, 50, 250)
gaussian_500 = np.concatenate([gaussian_1, gaussian_2])

# Uniform column
uniform_1 = np.random.uniform(0, 1, 250)
uniform_2 = np.random.uniform(3, 4, 250) 
uniform_500 = np.concatenate([uniform_1, uniform_2])

# Creating random label 
label = [0] * 500

# 250 random indices chosen from [0, 249] & [250, 500]
random_indices_1 = sample(range(0, 249, 1), 125)
random_indices_2 = sample(range(250, 500, 1), 125) 

# Assigned 1 to array based on those indices
for i in random_indices_1:
    label[i] = 1
    
for i in random_indices_2:
    label[i] = 1
    
# Create dataframe of data
data = {
    'Binomial': binomial_500,
    'Poisson:': poisson_500, 
    'Uniform': uniform_500,
    'Gaussian': gaussian_500,
    'Label': label
}

df = pd.DataFrame(data)    

# Filter the DataFrame to get rows where label=1
filtered_df_0 = df[df['Label'] == 0]
filtered_df_1 = df[df['Label'] == 1]

# Calculate the mean of the Binomial columns for where label=0 and label=1
mean_value_0 = filtered_df_0['Binomial'].mean()
mean_value_1 = filtered_df_1['Binomial'].mean()

# Running test
print(mean_value_0)
print(mean_value_1)
result1 = kruskal(filtered_df_0['Binomial'], filtered_df_1['Binomial']) # Output: Not Significant
result2 = f_oneway(filtered_df_0['Binomial'], filtered_df_1['Binomial']) # Output: Not Significant
print(result1)
print(result2)
print(ttest_ind(filtered_df_0['Binomial'], filtered_df_1['Binomial']))

####################### Re-making data to run tests ###########################

# Create empty lists to save significance counts for each combination
ANOVA_bin_gaus_sigs = []
ANOVA_bin_uni_sigs = []

KWH_bin_gaus_sigs = []
KWH_bin_uni_sigs = []

AG_bin_gaus_sigs = []
AG_bin_uni_sigs = []


for i in range(150):
    
    # Gaussian column
    gaussian_1 = np.random.normal(100, 50, 250)
    gaussian_2 = np.random.normal(13, 50, 250)
    gaussian_500 = np.concatenate([gaussian_1, gaussian_2])

    # Uniform column
    uniform_1 = np.random.uniform(0, 1, 250)
    uniform_2 = np.random.uniform(3, 4, 250) 
    uniform_500 = np.concatenate([uniform_1, uniform_2])

    # Creating random label 
    label = [0] * 500

    # 250 random indices chosen from [0, 249] & [250, 500]
    random_indices_1 = sample(range(0, 249, 1), 125)
    random_indices_2 = sample(range(250, 500, 1), 125) 

    # Assigned 1 to array based on those indices
    for i in random_indices_1:
        label[i] = 1
        
    for i in random_indices_2:
        label[i] = 1
        
    # Create dataframe of data
    data = {
        'Label': label
    }

    df = pd.DataFrame(data)  
    
    # Label v. Gaussian
    f_BG, p1_BG = f_oneway(df['Label'], gaussian_500)
    k_BG, p2_BG = kruskal(df['Label'], gaussian_500)
    p3_BG = getattr(alexandergovern(df['Label'], gaussian_500),'pvalue')
    
    # Label v. Uniform
    f_BU, p1_BU = f_oneway(df['Label'], uniform_500)
    k_BU, p2_BU = kruskal(df['Label'], uniform_500)
    p3_BU = getattr(alexandergovern(df['Label'], uniform_500),'pvalue')
    
    # Label v. Gaussian
    if p1_BG < 0.05:
        ANOVA_bin_gaus_sigs.append(True)

    else:
        ANOVA_bin_gaus_sigs.append(False)
    if p2_BG < 0.05:
        KWH_bin_gaus_sigs.append(True)

    else:
        KWH_bin_gaus_sigs.append(False)
    if p3_BG < 0.05:
        AG_bin_gaus_sigs.append(True)

    else:
        AG_bin_gaus_sigs.append(False)
        
    # Label v. Uniform
    if p1_BU < 0.05:
        ANOVA_bin_uni_sigs.append(True)

    else:
        ANOVA_bin_uni_sigs.append(False)
    if p2_BU < 0.05:
        KWH_bin_uni_sigs.append(True)

    else:
        KWH_bin_uni_sigs.append(False)
    if p3_BU < 0.05:
        AG_bin_uni_sigs.append(True)

    else:
        AG_bin_uni_sigs.append(False)

        
# Repeat the test 150 time each
# Goal: to get not significant primarily when generating random values; 
# ensuring that the test is correct


# Count the number of True and False values across combinations

########## ANOVA
ANOVA_bin_gaus_sigs_TRUE = sum(ANOVA_bin_gaus_sigs) # BIN & GAUSS
ANOVA_bin_gaus_sigs_FALSE = len(ANOVA_bin_gaus_sigs) - ANOVA_bin_gaus_sigs_TRUE

ANOVA_bin_uni_sigs_TRUE = sum(ANOVA_bin_uni_sigs) # BIN & UNIFORM
ANOVA_bin_uni_sigs_FALSE = len(ANOVA_bin_uni_sigs) - ANOVA_bin_uni_sigs_TRUE

########## KWH
KWH_bin_gaus_sigs_TRUE = sum(KWH_bin_gaus_sigs) # BIN & GAUSS
KWH_bin_gaus_sigs_FALSE = len(KWH_bin_gaus_sigs) - KWH_bin_gaus_sigs_TRUE

KWH_bin_uni_sigs_TRUE = sum(KWH_bin_uni_sigs) # BIN & UNIFORM
KWH_bin_uni_sigs_FALSE = len(KWH_bin_uni_sigs) - KWH_bin_uni_sigs_TRUE

########### AG TEST
AG_bin_gaus_sigs_TRUE = sum(AG_bin_gaus_sigs) # BIN & GAUSS
AG_bin_gaus_sigs_FALSE = len(AG_bin_gaus_sigs) - AG_bin_gaus_sigs_TRUE

AG_bin_uni_sigs_TRUE = sum(AG_bin_uni_sigs) # BIN & UNIFORM
AG_bin_uni_sigs_FALSE = len(AG_bin_uni_sigs) - AG_bin_uni_sigs_TRUE

# ################## Graphing ######################################################

# # Data
# categories = ['ANOVA', 'Kruskal-Wallis H Test', 'Alexander-Govern']
# bin_gauss_SIGS = [ANOVA_bin_gaus_sigs_TRUE, KWH_bin_gaus_sigs_TRUE, AG_bin_gaus_sigs_TRUE]
# bin_gauss_NO_SIGS = [ANOVA_bin_gaus_sigs_FALSE, KWH_bin_gaus_sigs_FALSE, AG_bin_gaus_sigs_FALSE]

# bin_uniform_SIGS = [ANOVA_bin_uni_sigs_TRUE, KWH_bin_uni_sigs_TRUE, AG_bin_uni_sigs_TRUE]
# bin_uniform_NO_SIGS = [ANOVA_bin_uni_sigs_FALSE, KWH_bin_uni_sigs_FALSE, AG_bin_uni_sigs_FALSE]

# # Number of categories
# n = len(categories)

# # X axis locations for the groups
# ind = np.arange(n)

# # Width of the bars
# width = 0.35

# ##################### PLOT 1 ############################################

# # Plotting
# fig, ax = plt.subplots()

# # Bars for Significant & Not Significant for each combo
# bar1 = ax.bar(ind - width/2, bin_gauss_SIGS, width, label='Significant', color='skyblue')
# bar2 = ax.bar(ind + width/2, bin_gauss_NO_SIGS, width, label='Not Significant', color='salmon')

# # Adding labels, title, and legend
# ax.set_xlabel('Tests')
# ax.set_ylabel('Significance Counts')
# ax.set_title('ANOVA v. KWH Test v. AG Test\n on RANDOM LABEL & GAUSSIAN columns')
# ax.set_xticks(ind)
# ax.set_xticklabels(categories)
# ax.legend()

# # Show plot
# plt.show() 

# ##################### PLOT 2 ############################################

# # Plotting
# fig, ax = plt.subplots()

# # Bars for Significant & Not Significant for each combo
# bar3 = ax.bar(ind - width/2, bin_uniform_SIGS, width, label='Significant', color='skyblue')
# bar4 = ax.bar(ind + width/2, bin_uniform_NO_SIGS, width, label='Not Significant', color='salmon')

# # Adding labels, title, and legend
# ax.set_xlabel('Tests')
# ax.set_ylabel('Significance Counts')
# ax.set_title('ANOVA v. KWH Test v. AG Test\n on RANDOM LABEL & UNIFORM columns')
# ax.set_xticks(ind)
# ax.set_xticklabels(categories)
# ax.legend()

# # Show plot
# plt.show() 
