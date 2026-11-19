import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from telco_churn.dataset import make_target, split_data


def main():
    df = pd.read_csv("data/raw/telco.csv")

    train_df, valid_df, test_df = split_data(df)

    first_features = ["tenure", "MonthlyCharges"]
    X_train, y_train = train_df[first_features], make_target(train_df)
    X_valid, y_valid = valid_df[first_features], make_target(valid_df)

    models = {
        "baseline_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_two_features": LogisticRegression(max_iter=1000),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        valid_predictions = model.predict(X_valid)
        print(f"{name}: validation accuracy {accuracy_score(y_valid, valid_predictions):.3f}")
        print("confusion matrix, rows = true [0=No, 1=Yes], columns = predicted [0, 1]:")
        print(confusion_matrix(y_valid, valid_predictions, labels=[0, 1]))
    return models


if __name__ == "__main__":
    main()
