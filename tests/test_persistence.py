import joblib
import numpy as np

from telco_churn.prediction import predict

THRESHOLD = 0.5


def test_reload_preserves_predictions(tmp_path, fitted_pipeline, sample_features):
    artifact = tmp_path / "pipeline.joblib"
    before_classes = fitted_pipeline.predict(sample_features)
    before_scores = fitted_pipeline.predict_proba(sample_features)

    joblib.dump(fitted_pipeline, artifact)
    reloaded = joblib.load(artifact)

    assert reloaded is not fitted_pipeline
    np.testing.assert_array_equal(reloaded.predict(sample_features), before_classes)
    np.testing.assert_allclose(reloaded.predict_proba(sample_features), before_scores, atol=1e-12)


def test_reloaded_artifact_is_the_complete_pipeline(tmp_path, fitted_pipeline, sample_features):
    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(fitted_pipeline, artifact)
    reloaded = joblib.load(artifact)

    assert list(reloaded.named_steps) == ["preprocessor", "classifier"]
    out_before = predict(fitted_pipeline, sample_features, THRESHOLD)
    out_after = predict(reloaded, sample_features, THRESHOLD)
    np.testing.assert_allclose(
        out_after["churn_probability"], out_before["churn_probability"], atol=1e-12
    )
    np.testing.assert_array_equal(out_after["churn_prediction"], out_before["churn_prediction"])


def test_reload_does_not_refit(tmp_path, fitted_pipeline, sample_features):
    artifact = tmp_path / "pipeline.joblib"
    joblib.dump(fitted_pipeline, artifact)
    reloaded = joblib.load(artifact)

    def numeric_medians(pipeline):
        num = pipeline.named_steps["preprocessor"].named_transformers_["num"]
        return num.named_steps["imputer"].statistics_.copy()

    learned = numeric_medians(reloaded)
    reloaded.predict(sample_features.head(3))

    np.testing.assert_array_equal(numeric_medians(reloaded), numeric_medians(fitted_pipeline))
    np.testing.assert_array_equal(numeric_medians(reloaded), learned)
