# ============================
# Predictions
# ============================
Y_pred = model.predict(X_test)

# ============================
# Evaluation
# ============================
print("\nAccuracy :", accuracy_score(Y_test, Y_pred))
print("Confusion Matrix :\n", confusion_matrix(Y_test, Y_pred))
print("\nClassification Report :\n", classification_report(Y_test, Y_pred))

# ============================
# Final Result
# ============================
print("\nFinal Conclusion:")
print(f"Train Accuracy: {train_score:.4f}")
print(f"Test Accuracy: {test_score:.4f}")
print(f"Cross Validation Accuracy: {cv_scores.mean():.4f}")


