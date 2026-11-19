import numpy as np

from telco_churn.prediction import predict

THRESHOLD = 0.5


def test_predict_proba_contract(fitted_pipeline, sample_features):
    probabilities = fitted_pipeline.predict_proba(sample_features)

    assert probabilities.shape == (len(sample_features), 2)
    assert list(fitted_pipeline.classes_) == [0, 1]
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


def test_churn_probability_is_the_positive_class_column(fitted_pipeline, sample_features):
    positive_column = list(fitted_pipeline.classes_).index(1)
    expected = fitted_pipeline.predict_proba(sample_features)[:, positive_column]

    out = predict(fitted_pipeline, sample_features, THRESHOLD)

    np.testing.assert_allclose(out["churn_probability"], expected)
    np.testing.assert_array_equal(out["churn_prediction"], (expected >= THRESHOLD).astype(int))


def test_churn_probability_points_towards_churners(
    fitted_pipeline, sample_features, sample_target
):
    out = predict(fitted_pipeline, sample_features, THRESHOLD)
    scores = out["churn_probability"].to_numpy()

    assert (
        scores[sample_target.to_numpy() == 1].mean() > scores[sample_target.to_numpy() == 0].mean()
    )


def test_one_output_row_per_customer(fitted_pipeline, sample_features):
    for size in (1, 5, len(sample_features)):
        batch = sample_features.head(size)
        out = predict(fitted_pipeline, batch, THRESHOLD)
        assert len(out) == size
        assert list(out.columns) == ["churn_probability", "churn_prediction"]
