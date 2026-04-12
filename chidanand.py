# ============================
# Cross Validation (FIXED)
# ============================
cv_scores = cross_val_score(model, X, Y, cv=5)

print("\nCross Validation Scores :", cv_scores)
print("Mean Cross Validation Accuracy :", cv_scores.mean())

# ============================
# Train model
# ============================
model.fit(X_train, Y_train)

# ============================
# Train & Test Score
# ============================
train_score = model.score(X_train, Y_train)
test_score = model.score(X_test, Y_test)

print("\nTrain Model Score :", train_score)
print("Test Model Score :", test_score)
