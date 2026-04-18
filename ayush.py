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

Member 2 : Soumya Prakash Mudasi
# ============================
# Data Preprocessing
# ============================

# Save PassengerId
test_ids = test_df[&#39;PassengerId&#39;]

# Drop unnecessary columns
train_df = train_df.drop([&#39;PassengerId&#39;, &#39;Name&#39;, &#39;Ticket&#39;, &#39;Cabin&#39;], axis=1)

test_df = test_df.drop([&#39;PassengerId&#39;, &#39;Name&#39;, &#39;Ticket&#39;, &#39;Cabin&#39;], axis=1)

# Fill missing values (IMPROVED)
for df in [train_df, test_df]:
    df[&#39;Age&#39;] = df[&#39;Age&#39;].fillna(df[&#39;Age&#39;].median())   # better than mean
    df[&#39;Embarked&#39;] = df[&#39;Embarked&#39;].fillna(df[&#39;Embarked&#39;].mode()[0])

test_df[&#39;Fare&#39;] = test_df[&#39;Fare&#39;].fillna(test_df[&#39;Fare&#39;].median())

Member 3 : Thakare Gaurav
# ============================
# FEATURE ENGINEERING (NEW)
# ============================
for df in [train_df, test_df]:
    df[&#39;FamilySize&#39;] = df[&#39;SibSp&#39;] + df[&#39;Parch&#39;] + 1
    df[&#39;IsAlone&#39;] = 1
    df.loc[df[&#39;FamilySize&#39;] &gt; 1, &#39;IsAlone&#39;] = 0

Member 4 : Makireddy Dinesh Reddy
# ============================
# Encoding (REPLACED LabelEncoder)
# ============================
train_df = pd.get_dummies(train_df, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)

# Align columns
train_df, test_df = train_df.align(test_df, join=&#39;left&#39;, axis=1, fill_value=0)

Member 5 : Ayush Basone
# ============================
# Split data
# ============================
X = train_df.drop(&#39;Survived&#39;, axis=1)
Y = train_df[&#39;Survived&#39;]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ============================
# Scaling
# ============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ============================
# Model (Improved Logistic Regression)
# ============================
model = LogisticRegression(max_iter=2000)

Member 6 : Chidanand Shankar Parashetti
# ============================
# Cross Validation (FIXED)

# ============================
cv_scores = cross_val_score(model, X, Y, cv=5)

print(&quot;\nCross Validation Scores :&quot;, cv_scores)
print(&quot;Mean Cross Validation Accuracy :&quot;, cv_scores.mean())

# ============================
# Train model
# ============================
model.fit(X_train, Y_train)

# ============================
# Train &amp; Test Score
# ============================
train_score = model.score(X_train, Y_train)
test_score = model.score(X_test, Y_test)

print(&quot;\nTrain Model Score :&quot;, train_score)
print(&quot;Test Model Score :&quot;, test_score)

Member 7 : Himarapura Kumarappa Gari
Manikanta
# ============================
# Predictions
# ============================
Y_pred = model.predict(X_test)

# ============================

# Evaluation
# ============================
print(&quot;\nAccuracy :&quot;, accuracy_score(Y_test, Y_pred))
print(&quot;Confusion Matrix :\n&quot;, confusion_matrix(Y_test, Y_pred))
print(&quot;\nClassification Report :\n&quot;, classification_report(Y_test, Y_pred))

# ============================
# Final Result
# ============================
print(&quot;\nFinal Conclusion:&quot;)
print(f&quot;Train Accuracy: {train_score:.4f}&quot;)
print(f&quot;Test Accuracy: {test_score:.4f}&quot;)
print(f&quot;Cross Validation Accuracy: {cv_scores.mean():.4f}&quot;)