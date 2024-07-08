import pandas as pd
from sklearn.model_selection import train_test_split
import sys

sys.path.append("C:/Users/kateh/OneDrive/Documents/GitHub/BP24")
import Measure_Patterns

################## Import data ################################################
df = pd.read_pickle("C:/Users/kateh/criminal_justice_data.pkl")

################## Data cleaning ##############################################
# Whatever changes you make to your data set

################## Split dataset into X_train and y_train #####################
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1], test_size=0.2, random_state=42)

# Call the measure_patterns function
Measure_Patterns.MeasurePatterns(X_train, y_train)