from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from telco_churn.dataset import load_data, make_target, split_data

numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = ["Contract", "InternetService", "PaymentMethod"]
features = numeric_features + categorical_features


def build_pipeline(classifier):
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


if __name__ == "__main__":
    df = load_data()
    train_df, valid_df, test_df = split_data(df)

    X_train, y_train = train_df[features], make_target(train_df)
    X_valid, y_valid = valid_df[features], make_target(valid_df)

    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X_train, y_train)

    valid_predictions = pipe.predict(X_valid)

    print("Validation accuracy:", round(accuracy_score(y_valid, valid_predictions), 3))
    print("Confusion matrix (rows = true, cols = predicted; order [0=No, 1=Yes]):")
    print(confusion_matrix(y_valid, valid_predictions, labels=[0, 1]))
