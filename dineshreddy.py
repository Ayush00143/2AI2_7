# ============================
# Encoding (REPLACED LabelEncoder)
# ============================
train_df = pd.get_dummies(train_df, drop_first=True)
test_df = pd.get_dummies(test_df, drop_first=True)

# Align columns
train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)
