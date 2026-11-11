"""Feature transformations for sparse and heavy-tailed market inputs."""

from typing import List, Optional, Sequence

import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler


SPARSE_FEATURES = ("X41", "X23")
LOG_FEATURES = ("X16", "X41", "X23")


def signed_log1p(values: np.ndarray) -> np.ndarray:
    """Compress signed values while preserving their sign and exact zeros."""
    return np.sign(values) * np.log1p(np.abs(values))


class MarketFeatureScaler:
    """Fit one of the supported feature transformations on training rows only."""

    METHODS = ("robust", "signed_log_standard", "sparse_signed_log")

    def __init__(self, feature_names: Sequence[str], method: str = "robust") -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown feature scaling method {method!r}; choose from {self.METHODS}.")
        self.feature_names = list(feature_names)
        self.method = method
        self.scaler: Optional[object] = None
        self.sparse_scalers = {}
        self.sparse_indices = {}
        self.feature_names_out = list(self.feature_names)

    def _transform_inputs(self, values: np.ndarray) -> np.ndarray:
        transformed = np.asarray(values, dtype=np.float64).copy()
        if transformed.ndim != 2 or transformed.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Expected a 2D array with {len(self.feature_names)} columns; "
                f"received shape {transformed.shape}."
            )
        if not np.isfinite(transformed).all():
            raise ValueError("Feature values must all be finite.")

        if self.method == "signed_log_standard":
            indices = [i for i, name in enumerate(self.feature_names) if name.startswith("X")]
            transformed[:, indices] = signed_log1p(transformed[:, indices])
        elif self.method == "sparse_signed_log":
            for name in LOG_FEATURES:
                if name in self.feature_names:
                    index = self.feature_names.index(name)
                    transformed[:, index] = signed_log1p(transformed[:, index])
        return transformed

    def fit(self, values: np.ndarray) -> "MarketFeatureScaler":
        transformed = self._transform_inputs(values)
        if self.method == "signed_log_standard":
            self.scaler = StandardScaler().fit(transformed)
        else:
            self.scaler = RobustScaler().fit(transformed)

        self.sparse_scalers = {}
        self.sparse_indices = {}
        self.feature_names_out = list(self.feature_names)
        if self.method == "sparse_signed_log":
            for name in SPARSE_FEATURES:
                if name not in self.feature_names:
                    continue
                index = self.feature_names.index(name)
                nonzero = transformed[:, index] != 0
                if not nonzero.any():
                    raise ValueError(f"Cannot scale {name}: no nonzero training observations.")
                self.sparse_scalers[name] = RobustScaler().fit(
                    transformed[nonzero][:, [index]]
                )
                self.sparse_indices[name] = index
                self.feature_names_out.append(f"{name}_is_nonzero")
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise ValueError("MarketFeatureScaler must be fitted before transform.")
        transformed = self._transform_inputs(values)
        scaled = self.scaler.transform(transformed)

        indicators = []
        for name, index in self.sparse_indices.items():
            nonzero = transformed[:, index] != 0
            scaled[:, index] = 0.0
            scaled[nonzero, index] = self.sparse_scalers[name].transform(
                transformed[nonzero][:, [index]]
            ).ravel()
            indicators.append(nonzero.astype(np.float64))
        if indicators:
            scaled = np.column_stack([scaled, *indicators])
        return scaled

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        return self.fit(values).transform(values)
