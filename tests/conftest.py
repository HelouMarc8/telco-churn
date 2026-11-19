import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from telco_churn.ml_preprocessing import build_pipeline, features

CUSTOMERS = [
    ("C01", 2, 85.0, 170.0, "Month-to-month", "Fiber optic", "Electronic check", 1),
    ("C02", 1, 95.5, 95.5, "Month-to-month", "Fiber optic", "Electronic check", 1),
    ("C03", 5, 79.9, 399.5, "Month-to-month", "Fiber optic", "Mailed check", 1),
    ("C04", 3, 70.2, 210.6, "Month-to-month", "DSL", "Electronic check", 1),
    ("C05", 8, 99.0, 792.0, "Month-to-month", "Fiber optic", "Electronic check", 1),
    ("C06", 12, 89.1, 1069.2, "One year", "Fiber optic", "Electronic check", 1),
    ("C07", 4, 55.0, 220.0, "Month-to-month", "DSL", "Mailed check", 0),
    ("C08", 60, 25.0, 1500.0, "Two year", "No", "Bank transfer (automatic)", 0),
    ("C09", 72, 110.0, 7920.0, "Two year", "Fiber optic", "Credit card (automatic)", 0),
    ("C10", 48, 60.5, 2904.0, "One year", "DSL", "Bank transfer (automatic)", 0),
    ("C11", 36, 20.0, 720.0, "Two year", "No", "Mailed check", 0),
    ("C12", 66, 75.3, 4969.8, "Two year", "DSL", "Credit card (automatic)", 0),
    ("C13", 24, 45.0, 1080.0, "One year", "DSL", "Credit card (automatic)", 0),
    ("C14", 52, 19.9, 1034.8, "Two year", "No", "Bank transfer (automatic)", 0),
    ("C15", 30, 65.0, 1950.0, "One year", "Fiber optic", "Bank transfer (automatic)", 0),
    ("C16", 18, 50.0, 900.0, "Month-to-month", "DSL", "Credit card (automatic)", 0),
]


@pytest.fixture
def sample_customers():
    return pd.DataFrame(
        CUSTOMERS,
        columns=["customerID", *features, "churn_label"],
    )


@pytest.fixture
def sample_features(sample_customers):
    return sample_customers[features].copy()


@pytest.fixture
def sample_target(sample_customers):
    return sample_customers["churn_label"].copy()


@pytest.fixture
def fitted_pipeline(sample_features, sample_target):
    pipeline = build_pipeline(LogisticRegression(max_iter=1000, random_state=0))
    return pipeline.fit(sample_features, sample_target)
