import numpy as np
import pytest

from src.data.feature_scaling import MarketFeatureScaler, signed_log1p


def test_signed_log1p_preserves_sign_and_zero():
    values = np.array([-100.0, 0.0, 100.0])

    transformed = signed_log1p(values)

    assert transformed[0] < 0
    assert transformed[1] == 0
    assert transformed[2] > 0
    np.testing.assert_allclose(transformed[0], -transformed[2])


@pytest.mark.parametrize("method", MarketFeatureScaler.METHODS)
def test_scaling_methods_fit_and_transform(method):
    features = np.array([
        [100.0, 0.0, 0.0, -1000.0],
        [101.0, 0.0, 10.0, 0.0],
        [102.0, 10.0, 20.0, 1000.0],
        [103.0, -10.0, 0.0, 2000.0],
    ])
    names = ["underlying", "X41", "X16", "X23"]
    scaler = MarketFeatureScaler(names, method=method)

    train_scaled = scaler.fit_transform(features[:3])
    test_scaled = scaler.transform(features[3:])

    assert np.isfinite(train_scaled).all()
    assert np.isfinite(test_scaled).all()
    assert train_scaled.shape[1] == (
        len(names) + 2 if method == "sparse_signed_log" else len(names)
    )


def test_sparse_scaling_keeps_zero_state_and_adds_indicators():
    features = np.array([
        [0.0, 0.0, -10.0],
        [0.0, 20.0, 0.0],
        [10.0, 0.0, 10.0],
        [20.0, -20.0, 100.0],
    ])
    scaler = MarketFeatureScaler(["X41", "X23", "X16"], method="sparse_signed_log")
    transformed = scaler.fit_transform(features)

    assert scaler.feature_names_out == [
        "X41", "X23", "X16", "X41_is_nonzero", "X23_is_nonzero"
    ]
    assert transformed[0, 0] == 0.0
    assert transformed[0, 3] == 0.0
    assert transformed[0, 4] == 0.0
    assert transformed[1, 4] == 1.0
    assert transformed[2, 3] == 1.0


def test_sparse_scaler_rejects_all_zero_training_feature():
    scaler = MarketFeatureScaler(["X41"], method="sparse_signed_log")

    with pytest.raises(ValueError, match="no nonzero training observations"):
        scaler.fit(np.zeros((4, 1)))
