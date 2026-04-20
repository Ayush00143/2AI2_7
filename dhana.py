# import library
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# load dataset
train_df = pd.read_csv(&quot;Titanic_train.csv&quot;)
test_df = pd.read_csv(&quot;Titanic_test.csv&quot;)

print(train_df.head())
print(train_df.isnull().sum())