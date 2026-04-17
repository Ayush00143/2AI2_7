import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming train_df is loaded elsewhere, e.g.:
# train_df = pd.read_csv('train.csv')

# ============================
# Split data
# ============================
X = train_df.drop('Survived', axis=1)
Y = train_df['Survived']

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
model.fit(X_train, Y_train)

# Predict
Y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(Y_test, Y_pred)
print(f'Accuracy: {accuracy}')