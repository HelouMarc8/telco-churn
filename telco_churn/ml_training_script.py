from sklearn.linear_model import LogisticRegression
import pandas as pd

from telco_churn.dataset import split_data

df = pd.read_csv("data/raw/telco.csv")

train_df, valid_df, test_df = split_data(df)

first_features = ["tenure", "MonthlyCharges"]
X_train = train_df[first_features]  # Ensure numeric types before fitting.
y_train = train_df["Churn"].map({"No": 0, "Yes": 1})
first_model = LogisticRegression(max_iter=1000)
first_model.fit(X_train, y_train)
# TODO: build X_valid/y_valid and obtain validation predictions.
X_valid = valid_df[first_features]  # Ensure numeric types before fitting.
y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})
valid_predictions = first_model.predict(X_valid)

# TODO: build X_test/y_test and obtain test predictions.
# X_test = test_df[first_features]  # Ensure numeric types before fitting.
# y_test = test_df["Churn"].map({"No": 0, "Yes": 1})
# test_predictions = first_model.predict(X_test)