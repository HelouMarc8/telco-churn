from sklearn.linear_model import LogisticRegression
import pandas as pd

from telco_churn.dataset import split_data

df = pd.read_csv("data/raw/telco.csv")

train_df, valid_df, test_df = split_data(df)

first_features = ["tenure", "MonthlyCharges"]
X_train = train_df[first_features]
y_train = train_df["Churn"].map({"No": 0, "Yes": 1})
first_model = LogisticRegression(max_iter=1000)
first_model.fit(X_train, y_train)
X_valid = valid_df[first_features]
y_valid = valid_df["Churn"].map({"No": 0, "Yes": 1})
valid_predictions = first_model.predict(X_valid)
