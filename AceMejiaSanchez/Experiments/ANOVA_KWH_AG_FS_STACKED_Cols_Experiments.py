# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 09:31:05 2024

@author: aceme
"""

################## Import relevant packages
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

################## Set Seed ####################################################

# Set a seed for reproducibility
np.random.seed(42)

################## Test Comparisons ############################################

# Combinations
# binomial_150, gaussian_150
# binomial_150, uniform_150
# poisson_150, gaussian_150
# poisson_150, uniform_150

################## Data #########################################

# Binomial column
n = 10
prob = .5
binomial_150 = np.random.binomial(n, prob, 500)

# Poisson column
lamda = 1
poisson_150 = np.random.poisson(lamda, 500)

# Gaussian column
gaussian_1 = np.random.normal(100, 50, 250)
gaussian_2 = np.random.normal(13, 50, 250)
gaussian_150 = np.concatenate([gaussian_1, gaussian_2])

# Uniform column
uniform_1 = np.random.uniform(0, 1, 250)
uniform_2 = np.random.uniform(3, 4, 250) 
uniform_150 = np.concatenate([uniform_1, uniform_2])

################## ANOVA #########################################

print("\n------------------ ANOVA (Feature vs Label) -----------------------")

### Testing consistency of ANOVA test

# Create empty lists to save significance counts for each combination
ANOVA_bin_gaus_sigs = []
ANOVA_bin_uni_sigs = []
ANOVA_pois_gaus_sigs = []
ANOVA_pois_uni_sigs = []

# Create empty dictionary to safve final list for each combination
ANOVA_sigs = []

for i in range(150):  

    # Run test for different combinations
    f1, p1 = f_oneway(binomial_150, gaussian_150)
    f2, p2 = f_oneway(binomial_150, uniform_150)
    f3, p3 = f_oneway(poisson_150, gaussian_150)
    f4, p4 = f_oneway(poisson_150, uniform_150)
    
    # Save significance counts for each combination in a list
    if p1 < 0.05:
        ANOVA_bin_gaus_sigs.append(True)
    else:
        ANOVA_bin_gaus_sigs.append(False)
    
    if p2 < 0.05:
        ANOVA_bin_uni_sigs.append(True)
    else:
        ANOVA_bin_uni_sigs.append(False)
    
    if p3 < 0.05:
        ANOVA_pois_gaus_sigs.append(True)
    else:
        ANOVA_pois_gaus_sigs.append(False)
    
    if p4 < 0.05:
        ANOVA_pois_uni_sigs.append(True)
    else:
        ANOVA_pois_uni_sigs.append(False)

# Count the number of True and False values across combinations
ANOVA_bin_gaus_sigs_TRUE = sum(ANOVA_bin_gaus_sigs) # BIN & GAUSS
ANOVA_bin_gaus_sigs_FALSE = len(ANOVA_bin_gaus_sigs) - ANOVA_bin_gaus_sigs_TRUE

ANOVA_bin_uni_sigs_TRUE = sum(ANOVA_bin_uni_sigs) # BIN & UNIFORM
ANOVA_bin_uni_sigs_FALSE = len(ANOVA_bin_uni_sigs) - ANOVA_bin_uni_sigs_TRUE

ANOVA_pois_gaus_sigs_TRUE = sum(ANOVA_pois_gaus_sigs) # POISSON & GAUSS
ANOVA_pois_gaus_sigs_FALSE = len(ANOVA_pois_gaus_sigs) - ANOVA_pois_gaus_sigs_TRUE

ANOVA_pois_uni_sigs_TRUE = sum(ANOVA_pois_uni_sigs) # POISSON & GAUSS
ANOVA_pois_uni_sigs_FALSE = len(ANOVA_pois_uni_sigs) - ANOVA_pois_uni_sigs_TRUE

# Print counts for each combination
print("Number of True values for Binomial & Gaussian:", ANOVA_bin_gaus_sigs_TRUE)
print("Number of False values for Binomial & Gaussian:", ANOVA_bin_gaus_sigs_FALSE)

print("Number of True values for Binomial & Uniform:", ANOVA_bin_uni_sigs_TRUE)
print("Number of False values for Binomial & Uniform:", ANOVA_bin_uni_sigs_FALSE)

print("Number of True values for Poisson & Gaussian:", ANOVA_pois_gaus_sigs_TRUE)
print("Number of False values for Poisson & Gaussian:", ANOVA_pois_gaus_sigs_FALSE)

print("Number of True values for Poisson & Uniform:", ANOVA_pois_uni_sigs_TRUE)
print("Number of False values for Poisson & Uniform:", ANOVA_pois_uni_sigs_FALSE)

# # CODE CHECK: Save significance counts list DICTIONARY for each combination
# ANOVA_sigs.append({
#     "binomial_150 & gaussian_150": ANOVA_bin_gaus_sigs, 
#     "binomial_150 & uniform_150": ANOVA_bin_uni_sigs, 
#     "poisson_150 & gaussian_150": ANOVA_pois_gaus_sigs, 
#     "poisson_150 & uniform_150": ANOVA_pois_uni_sigs})    

# from pprint import pprint
# pprint(ANOVA_sigs)

################## KRUSKAL-WALLIS H Test (FvL) #######################################

print("\n------------------ Kruskal-Wallis H Test (Feature vs Label) -----------------------")

### Testing consistency of ANOVA test

# Create empty lists to save significance counts for each combination
KWH_bin_gaus_sigs = []
KWH_bin_uni_sigs = []
KWH_pois_gaus_sigs = []
KWH_pois_uni_sigs = []

# Create empty dictionary to save final list for each combination
KWH_sigs = []

# Create an empty list to save sample values for scatterplot
KWH_scatterplot_SIG_pois_gaus = []
KWH_scatterplot_NO_SIG_pois_gaus = []

KWH_scatterplot_SIG_pois_uniform = []
KWH_scatterplot_NO_SIG_pois_uniform = []


for i in range(150):  
    
    # Run test for different combinations
    k1, p1 = kruskal(binomial_150, gaussian_150)
    k2, p2 = kruskal(binomial_150, uniform_150)
    k3, p3 = kruskal(poisson_150, gaussian_150)
    k4, p4 = kruskal(poisson_150, uniform_150)
    
    # Save significance counts for each combination in a list
    if p1 < 0.05:
        KWH_bin_gaus_sigs.append(True)

    else:
        KWH_bin_gaus_sigs.append(False)
    
    if p2 < 0.05:
        KWH_bin_uni_sigs.append(True)
    else:
        KWH_bin_uni_sigs.append(False)
    
    if p3 < 0.05:
        KWH_pois_gaus_sigs.append(True)
        KWH_scatterplot_SIG_pois_gaus.append({"RUN #": i, 
                                "Poisson 150": poisson_150, 
                                "Gaussian 150": gaussian_150})
    else:
        KWH_pois_gaus_sigs.append(False)
        KWH_scatterplot_NO_SIG_pois_gaus.append({"RUN #": i, 
                                "Poisson 150": poisson_150, 
                                "Gaussian 150": gaussian_150})
    
    if p4 < 0.05:
        KWH_pois_uni_sigs.append(True)
        KWH_scatterplot_SIG_pois_uniform.append({"RUN #": i, 
                                "Poisson 150": poisson_150, 
                                "Uniform 150": uniform_150})
    else:
        KWH_pois_uni_sigs.append(False)
        KWH_scatterplot_NO_SIG_pois_uniform.append({"RUN #": i, 
                                "Poisson 150": poisson_150, 
                                "Uniform 150": uniform_150})

# Count the number of True and False values across combinations
KWH_bin_gaus_sigs_TRUE = sum(KWH_bin_gaus_sigs) # BIN & GAUSS
KWH_bin_gaus_sigs_FALSE = len(KWH_bin_gaus_sigs) - KWH_bin_gaus_sigs_TRUE

KWH_bin_uni_sigs_TRUE = sum(KWH_bin_uni_sigs) # BIN & UNIFORM
KWH_bin_uni_sigs_FALSE = len(KWH_bin_uni_sigs) - KWH_bin_uni_sigs_TRUE

KWH_pois_gaus_sigs_TRUE = sum(KWH_pois_gaus_sigs) # POISSON & GAUSS
KWH_pois_gaus_sigs_FALSE = len(KWH_pois_gaus_sigs) - KWH_pois_gaus_sigs_TRUE

KWH_pois_uni_sigs_TRUE = sum(KWH_pois_uni_sigs) # POISSON & GAUSS
KWH_pois_uni_sigs_FALSE = len(KWH_pois_uni_sigs) - KWH_pois_uni_sigs_TRUE

# Print counts for each combination
print("Number of True values for Binomial & Gaussian:", KWH_bin_gaus_sigs_TRUE)
print("Number of False values for Binomial & Gaussian:", KWH_bin_gaus_sigs_FALSE)

print("Number of True values for Binomial & Uniform:", KWH_bin_uni_sigs_TRUE)
print("Number of False values for Binomial & Uniform:", KWH_bin_uni_sigs_FALSE)

print("Number of True values for Poisson & Gaussian:", KWH_pois_gaus_sigs_TRUE)
print("Number of False values for Poisson & Gaussian:", KWH_pois_gaus_sigs_FALSE)

print("Number of True values for Poisson & Uniform:", KWH_pois_uni_sigs_TRUE)
print("Number of False values for Poisson & Uniform:", KWH_pois_uni_sigs_FALSE)

# # CODE CHECK: Save significance counts list DICTIONARY for each combination
# KWH_sigs.append({
#     "binomial_150 & gaussian_150": KWH_bin_gaus_sigs, 
#     "binomial_150 & uniform_150": KWH_bin_uni_sigs, 
#     "poisson_150 & gaussian_150": KWH_pois_gaus_sigs, 
#     "poisson_150 & uniform_150": KWH_pois_uni_sigs})    

# from pprint import pprint
# pprint(KWH_sigs)

################## ALEXANDER-GOVERN Test #######################################

print("\n------------------ ALEXANDER-GOVERN Test (Feature vs Label) -----------------------")

### Testing consistency of ANOVA test

# Create empty lists to save significance counts for each combination
AG_bin_gaus_sigs = []
AG_bin_uni_sigs = []
AG_pois_gaus_sigs = []
AG_pois_uni_sigs = []

# Create empty dictionary to safve final list for each combination
AG_sigs = []

for i in range(150):  
    
    # Run test for different combinations & use getattr to select p-value 
    p1 = getattr(alexandergovern(binomial_150, gaussian_150),'pvalue')
    p2 = getattr(alexandergovern(binomial_150, uniform_150),'pvalue')
    p3 = getattr(alexandergovern(poisson_150, gaussian_150),'pvalue')
    p4 = getattr(alexandergovern(poisson_150, uniform_150),'pvalue')
    
    # Save significance counts for each combination in a list
    if p1 < 0.05:
        AG_bin_gaus_sigs.append(True)
    else:
        AG_bin_gaus_sigs.append(False)
    
    if p2 < 0.05:
        AG_bin_uni_sigs.append(True)
    else:
        AG_bin_uni_sigs.append(False)
    
    if p3 < 0.05:
        AG_pois_gaus_sigs.append(True)
    else:
        AG_pois_gaus_sigs.append(False)
    
    if p4 < 0.05:
        AG_pois_uni_sigs.append(True)
    else:
        AG_pois_uni_sigs.append(False)

# Count the number of True and False values across combinations
AG_bin_gaus_sigs_TRUE = sum(AG_bin_gaus_sigs) # BIN & GAUSS
AG_bin_gaus_sigs_FALSE = len(AG_bin_gaus_sigs) - AG_bin_gaus_sigs_TRUE

AG_bin_uni_sigs_TRUE = sum(AG_bin_uni_sigs) # BIN & UNIFORM
AG_bin_uni_sigs_FALSE = len(AG_bin_uni_sigs) - AG_bin_uni_sigs_TRUE

AG_pois_gaus_sigs_TRUE = sum(AG_pois_gaus_sigs) # POISSON & GAUSS
AG_pois_gaus_sigs_FALSE = len(AG_pois_gaus_sigs) - AG_pois_gaus_sigs_TRUE

AG_pois_uni_sigs_TRUE = sum(AG_pois_uni_sigs) # POISSON & GAUSS
AG_pois_uni_sigs_FALSE = len(AG_pois_uni_sigs) - AG_pois_uni_sigs_TRUE

# Print counts for each combination
print("Number of True values for Binomial & Gaussian:", AG_bin_gaus_sigs_TRUE)
print("Number of False values for Binomial & Gaussian:", AG_bin_gaus_sigs_FALSE)

print("Number of True values for Binomial & Uniform:", AG_bin_uni_sigs_TRUE)
print("Number of False values for Binomial & Uniform:", AG_bin_uni_sigs_FALSE)

print("Number of True values for Poisson & Gaussian:", AG_pois_gaus_sigs_TRUE)
print("Number of False values for Poisson & Gaussian:", AG_pois_gaus_sigs_FALSE)

print("Number of True values for Poisson & Uniform:", AG_pois_uni_sigs_TRUE)
print("Number of False values for Poisson & Uniform:", AG_pois_uni_sigs_FALSE)

# # CODE CHECK: Save significance counts list DICTIONARY for each combination
# ANOVA_sigs.append({
#     "binomial_150 & gaussian_150": ANOVA_bin_gaus_sigs, 
#     "binomial_150 & uniform_150": ANOVA_bin_uni_sigs, 
#     "poisson_150 & gaussian_150": ANOVA_pois_gaus_sigs, 
#     "poisson_150 & uniform_150": ANOVA_pois_uni_sigs})    

# from pprint import pprint
# pprint(ANOVA_sigs)

################## FRIEDMAN SQUARED #######################################

print("\n------------------ Friedman Squared Test (Feature vs Label) -----------------------")

### Testing consistency of ANOVA test

# Create empty lists to save significance counts for each combination
FS_bin_gaus_sigs = []
FS_bin_uni_sigs = []
FS_pois_gaus_sigs = []
FS_pois_uni_sigs = []

# Create empty dictionary to save final list for each combination
FS_sigs = []

# Create an empty list to save sample values for scatterplot
FS_scatterplot_SIG_pois_gaus = []
FS_scatterplot_NO_SIG_pois_gaus = []

FS_scatterplot_SIG_pois_uniform = []
FS_scatterplot_NO_SIG_pois_uniform = []


for i in range(150):  
    
    # Run test for different combinations
    k1, p1 = friedmanchisquare(binomial_150, gaussian_150, poisson_150)
    k2, p2 = friedmanchisquare(binomial_150, uniform_150, poisson_150)
    k3, p3 = friedmanchisquare(poisson_150, gaussian_150, poisson_150)
    k4, p4 = friedmanchisquare(poisson_150, uniform_150, poisson_150)
    
    # Save significance counts for each combination in a list
    if p1 < 0.05:
        FS_bin_gaus_sigs.append(True)
    else:
        FS_bin_gaus_sigs.append(False)
    
    if p2 < 0.05:
        FS_bin_uni_sigs.append(True)
    else:
        FS_bin_uni_sigs.append(False)
    
    if p3 < 0.05:
        FS_pois_gaus_sigs.append(True)

    else:
        FS_pois_gaus_sigs.append(False)

    
    if p4 < 0.05:
        FS_pois_uni_sigs.append(True)

    else:
        FS_pois_uni_sigs.append(False)


# Count the number of True and False values across combinations
FS_bin_gaus_sigs_TRUE = sum(FS_bin_gaus_sigs) # BIN & GAUSS
FS_bin_gaus_sigs_FALSE = len(FS_bin_gaus_sigs) - FS_bin_gaus_sigs_TRUE

FS_bin_uni_sigs_TRUE = sum(FS_bin_uni_sigs) # BIN & UNIFORM
FS_bin_uni_sigs_FALSE = len(FS_bin_uni_sigs) - FS_bin_uni_sigs_TRUE

FS_pois_gaus_sigs_TRUE = sum(FS_pois_gaus_sigs) # POISSON & GAUSS
FS_pois_gaus_sigs_FALSE = len(FS_pois_gaus_sigs) - FS_pois_gaus_sigs_TRUE

FS_pois_uni_sigs_TRUE = sum(FS_pois_uni_sigs) # POISSON & GAUSS
FS_pois_uni_sigs_FALSE = len(FS_pois_uni_sigs) - FS_pois_uni_sigs_TRUE

# Print counts for each combination
print("Number of True values for Binomial & Gaussian:", FS_bin_gaus_sigs_TRUE)
print("Number of False values for Binomial & Gaussian:", FS_bin_gaus_sigs_FALSE)

print("Number of True values for Binomial & Uniform:", FS_bin_uni_sigs_TRUE)
print("Number of False values for Binomial & Uniform:", FS_bin_uni_sigs_FALSE)

print("Number of True values for Poisson & Gaussian:", FS_pois_gaus_sigs_TRUE)
print("Number of False values for Poisson & Gaussian:", FS_pois_gaus_sigs_FALSE)

print("Number of True values for Poisson & Uniform:", FS_pois_uni_sigs_TRUE)
print("Number of False values for Poisson & Uniform:", FS_pois_uni_sigs_FALSE)

# # CODE CHECK: Save significance counts list DICTIONARY for each combination
# FS_sigs.append({
#     "binomial_150 & gaussian_150": FS_bin_gaus_sigs, 
#     "binomial_150 & uniform_150": FS_bin_uni_sigs, 
#     "poisson_150 & gaussian_150": FS_pois_gaus_sigs, 
#     "poisson_150 & uniform_150": FS_pois_uni_sigs})    

# from pprint import pprint
# pprint(FS_sigs)

################## Graphing ######################################################

# Data
categories = ['ANOVA', 'Kruskal-Wallis H Test', 'Alexander-Govern', 'Friedman Squared']
bin_gauss_SIGS = [ANOVA_bin_gaus_sigs_TRUE, KWH_bin_gaus_sigs_TRUE, AG_bin_gaus_sigs_TRUE, FS_bin_gaus_sigs_TRUE]
bin_gauss_NO_SIGS = [ANOVA_bin_gaus_sigs_FALSE, KWH_bin_gaus_sigs_FALSE, AG_bin_gaus_sigs_FALSE, FS_bin_gaus_sigs_FALSE]

bin_uniform_SIGS = [ANOVA_bin_uni_sigs_TRUE, KWH_bin_uni_sigs_TRUE, AG_bin_uni_sigs_TRUE, FS_bin_uni_sigs_TRUE]
bin_uniform_NO_SIGS = [ANOVA_bin_uni_sigs_FALSE, KWH_bin_uni_sigs_FALSE, AG_bin_uni_sigs_FALSE, FS_bin_uni_sigs_FALSE]

poisson_gauss_SIGS = [ANOVA_pois_gaus_sigs_TRUE, KWH_pois_gaus_sigs_TRUE, AG_pois_gaus_sigs_TRUE, FS_pois_gaus_sigs_TRUE]
poisson_gauss_NO_SIGS = [ANOVA_pois_gaus_sigs_FALSE, KWH_pois_gaus_sigs_FALSE, AG_pois_gaus_sigs_FALSE, FS_pois_gaus_sigs_FALSE]

poisson_uniform_SIGS = [ANOVA_pois_uni_sigs_TRUE, KWH_pois_uni_sigs_TRUE, AG_pois_uni_sigs_TRUE, FS_pois_uni_sigs_TRUE]
poisson_uniform_NO_SIGS = [ANOVA_pois_uni_sigs_FALSE, KWH_pois_uni_sigs_FALSE, AG_pois_uni_sigs_FALSE, FS_pois_uni_sigs_FALSE]

# Number of categories
n = len(categories)

# X axis locations for the groups
ind = np.arange(n)

# Width of the bars
width = 0.35

##################### PLOT 1 ############################################

# Plotting
fig, ax = plt.subplots()

# Bars for Significant & Not Significant for each combo
bar1 = ax.bar(ind - width/2, bin_gauss_SIGS, width, label='Significant', color='skyblue')
bar2 = ax.bar(ind + width/2, bin_gauss_NO_SIGS, width, label='Not Significant', color='salmon')

# Adding labels, title, and legend
ax.set_xlabel('Tests')
ax.set_ylabel('Significance Counts')
ax.set_title('ANOVA v. KWH Test v. AG Test\n on BINOMIAL & GAUSSIAN columns')
ax.set_xticks(ind)
ax.set_xticklabels(categories)
ax.legend()

# Show plot
plt.show() 

##################### PLOT 2 ############################################

# Plotting
fig, ax = plt.subplots()

# Bars for Significant & Not Significant for each combo
bar3 = ax.bar(ind - width/2, bin_uniform_SIGS, width, label='Significant', color='skyblue')
bar4 = ax.bar(ind + width/2, bin_uniform_NO_SIGS, width, label='Not Significant', color='salmon')

# Adding labels, title, and legend
ax.set_xlabel('Tests')
ax.set_ylabel('Significance Counts')
ax.set_title('ANOVA v. KWH Test v. AG Test\n on BINOMIAL & UNIFORM columns')
ax.set_xticks(ind)
ax.set_xticklabels(categories)
ax.legend()

# Show plot
plt.show() 

##################### PLOT 3 ############################################

# Plotting
fig, ax = plt.subplots()

# Bars for Significant & Not Significant for each combo
bar5 = ax.bar(ind - width/2, poisson_gauss_SIGS, width, label='Significant', color='skyblue')
bar6 = ax.bar(ind + width/2, poisson_gauss_NO_SIGS, width, label='Not Significant', color='salmon')

# Adding labels, title, and legend
ax.set_xlabel('Tests')
ax.set_ylabel('Significance Counts')
ax.set_title('ANOVA v. KWH Test v. AG Test\n on POISSON & GAUSSIAN columns')
ax.set_xticks(ind)
ax.set_xticklabels(categories)
ax.legend()

# Show plot
plt.show() 

##################### PLOT 4 ############################################

# Plotting
fig, ax = plt.subplots()

# Bars for Significant & Not Significant for each combo
bar7 = ax.bar(ind - width/2, poisson_uniform_SIGS, width, label='Significant', color='skyblue')
bar8 = ax.bar(ind + width/2, poisson_uniform_NO_SIGS, width, label='Not Significant', color='salmon')

# Adding labels, title, and legend
ax.set_xlabel('Tests')
ax.set_ylabel('Significance Counts')
ax.set_title('ANOVA v. KWH Test v. AG Test\n on POISSON & UNIFORM columns')
ax.set_xticks(ind)
ax.set_xticklabels(categories)
ax.legend()

# Show plot
plt.show() 


##################### SCATTERPLOT 4 ############################################

### Creating Scatterplot

# # Reviewing saved values
# from pprint import pprint
# pprint(KWH_scatterplot)

# # Assigning variables: KWH "Significant"
# x = KWH_scatterplot_SIG_pois_uniform[0]['Uniform 150'] # Uniform var
# y = KWH_scatterplot_SIG_pois_uniform[1]['Uniform 150'] # Uniform var
# z = KWH_scatterplot_SIG_pois_uniform[0]['Poisson 150'] # Poisson var

# Assigning variables: KWH "Significant"
x = KWH_scatterplot_SIG_pois_gaus[0]['Gaussian 150'] # Gaussian var
y = KWH_scatterplot_SIG_pois_gaus[1]['Gaussian 150'] # Gaussian var
z = KWH_scatterplot_SIG_pois_gaus[0]['Poisson 150'] # Poisson var

# # Assigning variables: KWH "Not Significant"
# x = KWH_scatterplot_NO_SIG_pois_uniform[0]['Uniform 150'] # Uniform var
# y = KWH_scatterplot_NO_SIG_pois_uniform[1]['Uniform 150'] # Uniform var
# z = KWH_scatterplot_NO_SIG_pois_uniform[0]['Poisson 150'] # Poisson var

plt.figure(figsize=(8, 6))
scatter = plt.scatter(x, y, c=z, cmap='viridis', edgecolors="black", alpha=0.75)
#plt.colorbar(scatter, label='Intensity')  # Add colorbar indicating intensity

plt.title('Scatter plot comparing UNIFORM/GAUSSIAN Columns v. POISSON LABEL')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)

plt.show()
