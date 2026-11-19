import numpy as np
import pandas as pd
import pytest

from telco_churn.dataset import make_target
from telco_churn.ml_preprocessing import categorical_features, features, numeric_features
from telco_churn.prediction import predict

THRESHOLD = 0.5


def test_missing_required_feature_is_rejected(fitted_pipeline, sample_features):
    broken = sample_features.drop(columns=["Contract"])

    with pytest.raises(ValueError, match=r"Missing required feature columns: \['Contract'\]"):
        predict(fitted_pipeline, broken, THRESHOLD)


@pytest.mark.parametrize("dropped", numeric_features + categorical_features)
def test_each_required_feature_is_named_when_missing(fitted_pipeline, sample_features, dropped):
    broken = sample_features.drop(columns=[dropped])

    with pytest.raises(ValueError, match=f"Missing required feature columns: \\['{dropped}'\\]"):
        predict(fitted_pipeline, broken, THRESHOLD)


def test_target_and_identifier_are_not_model_inputs(fitted_pipeline, sample_customers):
    assert "Churn" not in features and "customerID" not in features
    assert list(fitted_pipeline.feature_names_in_) == features

    with_extras = sample_customers.assign(Churn="Yes")
    features_only = sample_customers[features]

    out_with_extras = predict(fitted_pipeline, with_extras, THRESHOLD)
    out_features_only = predict(fitted_pipeline, features_only, THRESHOLD)
    np.testing.assert_array_equal(
        out_with_extras["churn_probability"], out_features_only["churn_probability"]
    )
    assert list(out_with_extras["customerID"]) == list(sample_customers["customerID"])


def test_missing_numeric_and_unseen_category_still_predict(fitted_pipeline, sample_features):
    unseen = "Satellite"
    assert unseen not in set(sample_features["InternetService"])
    encoder = fitted_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
    learned = encoder.named_steps["encoder"].categories_[
        categorical_features.index("InternetService")
    ]
    assert unseen not in set(learned)

    varied = sample_features.copy()
    varied = varied.astype({"TotalCharges": "object"})
    varied.loc[0, "TotalCharges"] = " "
    varied.loc[1, "tenure"] = np.nan
    varied.loc[2, "InternetService"] = unseen

    out = predict(fitted_pipeline, varied, THRESHOLD)

    assert len(out) == len(varied)
    assert out.index.equals(varied.index)
    assert np.isfinite(out["churn_probability"]).all()
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["churn_prediction"]) <= {0, 1}


def test_malformed_numeric_text_is_rejected_not_imputed(fitted_pipeline, sample_features):
    malformed = sample_features.astype({"TotalCharges": "object"})
    malformed.loc[0, "TotalCharges"] = "abc"

    with pytest.raises(ValueError, match="abc"):
        predict(fitted_pipeline, malformed, THRESHOLD)


def test_target_mapping_makes_yes_the_positive_class():
    labels = pd.DataFrame({"Churn": ["Yes", "No", "No", "Yes"]})

    assert make_target(labels).tolist() == [1, 0, 0, 1]


def test_unexpected_target_label_is_rejected():
    with pytest.raises(ValueError, match="Unexpected Churn labels"):
        make_target(pd.DataFrame({"Churn": ["Yes", "Maybe"]}))
