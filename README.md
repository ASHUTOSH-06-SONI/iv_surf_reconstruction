# Implied Volatility Surface Reconstruction

Research code for reconstructing missing implied-volatility (IV) quotes from
partially observed option surfaces. The current benchmark uses historical
NIFTY 50 option data from a 2025 competition. The research goal is to compare
financial and data-driven reconstruction methods, then test how well they
generalize across more recent periods and other asset classes.

> **Current status:** the first retrospective, masked-quote benchmark is
> implemented. Linear interpolation and a neural surface model are promising
> on this dataset, but the result is preliminary and does not establish
> cross-asset generalization or arbitrage-free behavior.

## Research question

Given a single option surface with some observed call and put IV quotes, how
accurately can a method reconstruct the hidden quotes on that same surface?

The primary evaluation is **masked-quote completion**:

1. Start with a complete historical surface.
2. Keep a subset of quotes visible as model inputs.
3. Hide the remaining quotes.
4. Predict only the hidden quotes and compare with their known values.

This is different from ordinary row-wise IV prediction: reconstruction methods
must use the observed quotes from the surface being completed.

## Current dataset

The local competition dataset is stored under `data/nk-iv-prediction/` and is
ignored by Git.

| File | Contents |
|---|---|
| `train_data.parquet` | 178,340 historical rows; complete call/put IV surfaces; 42 anonymized market features |
| `test_data.parquet` | 12,065 partially observed surfaces; masked timestamps are unique IDs; expiry is omitted |
| `sample_submission.csv` | Competition-format output template |

The training period runs from **April 22 to May 7, 2025 (UTC)** and contains
three weekly expiries: April 24, April 30, and May 8. The training surface has
52 IV columns: 26 calls and 26 puts.

The provided test surface has about 60% of its IV entries missing. It has a
different strike grid from training: 42 strikes overlap, while 10 outer test
strikes have no training labels. Offline metrics therefore cover only the
shared 42 strikes. The masked test timestamps are IDs, not chronological time.

The data also has 42 anonymous `X0`–`X41` features. Their meanings and units
are unavailable. Some are sparse or have heavy tails, and feature distributions
can differ between train and test. Do not assume that an extreme value is
erroneous solely because it is large.

## Current masked-quote benchmark

Run from the repository root:

```bash
python -m src.evaluate.masked_quote_benchmark 2>&1 | tee results/masked_quote_benchmark.log
```

Default run settings are 12,000 sampled training surfaces, 500 validation
surfaces, 500 later holdout surfaces, 40 epochs for each neural model, and a
fixed random seed of 42. The visible call/put quote counts are sampled to
resemble those in the competition test data; the positions of visible strikes
are randomly selected.

Methods compared:

- **Per-surface median:** simple reference baseline using only that surface’s
  observed quotes.
- **Linear interpolation:** interpolates within call and put smiles separately.
- **SABR:** fits a SABR smile to observed quotes on each surface, with beta fixed.
- **Surface MLP:** unconstrained neural surface-completion model.
- **PINN:** the repository `PINNModel` used as a surface-completion network,
  with Softplus output and an IV-range physics penalty.

The benchmark PINN and MLP receive visible IV values, an observation mask,
log-moneyness, and expiry information; they do **not** currently use the
anonymous `X0`–`X41` features. They output a vector of IVs for the surface. The
neural models are trained on sampled surfaces from the earlier training period
and evaluated on later timestamps. In this experiment, “physics-informed”
means positive outputs and a soft penalty for predictions outside `[0, 3]`;
the benchmark does not enforce a pricing PDE, smoothness, or static
arbitrage-free constraints.

The separate `src/train/train_and_save.py` pipeline is a standard
feature-to-single-IV predictor with timestamp-grouped splitting and configurable
feature scaling. It is not the masked-surface model used for the results below.

### Results currently recorded

Metrics are decimal IV units; lower MAE/RMSE is better. These are from the default 500-surface validation and holdout run in
`results/masked_quote_benchmark.csv`.

| Split | Method | Hidden quotes | MAE | RMSE | R² |
|---|---|---:|---:|---:|---:|
| Validation | Surface median | 12,517 | 0.0671 | 0.0828 | -0.196 |
| Validation | Linear interpolation | 12,517 | **0.0109** | 0.0210 | 0.923 |
| Validation | SABR | 12,517 | 0.0514 | 0.0627 | 0.313 |
| Validation | Surface MLP | 12,517 | 0.0143 | **0.0177** | 0.945 |
| Validation | PINN | 12,517 | 0.0144 | 0.0179 | 0.944 |
| Holdout | Surface median | 12,699 | 0.1020 | 0.1256 | -0.164 |
| Holdout | Linear interpolation | 12,699 | 0.0182 | 0.0354 | 0.907 |
| Holdout | SABR | 12,699 | 0.0905 | 0.1111 | 0.089 |
| Holdout | Surface MLP | 12,699 | 0.0153 | 0.0197 | 0.971 |
| Holdout | PINN | 12,699 | **0.0152** | **0.0191** | **0.973** |

On this sampled holdout, the PINN and MLP beat the classical methods on RMSE;
linear interpolation has slightly lower validation MAE. This is one
retrospective dataset and one primary mask seed, so treat the ranking as
preliminary. The competition test set has no local ground truth available, so
these offline scores do not measure error on its hidden quotes.

Outputs:

- `results/masked_quote_benchmark.csv` — validation and holdout metrics
- `results/masked_quote_reconstruction.png` — observed, hidden, and predicted
  call/put IVs for an example surface
- `results/masked_quote_benchmark.log` — terminal output, if using the command
  above

To adjust run size, epochs, or seed:

```bash
python -m src.evaluate.masked_quote_benchmark \
  --train-rows 12000 \
  --evaluation-rows 500 \
  --epochs 40 \
  --seed 42 \
  --output results/masked_quote_benchmark.csv
```

## Feature-scaling experiment

`src.evaluate.scaling_comparison` compares input scalers for ordinary IV
prediction from market features. It is an auxiliary diagnostic, **not** the
masked-quote reconstruction benchmark, and its metrics should not be used to
choose the reconstruction model.

```bash
python -m src.evaluate.scaling_comparison --config config/default.yaml
```

Available feature scaling options for the training data pipeline:

- `robust` — RobustScaler, the configured default.
- `signed_log_standard` — signed `log1p` on `X` fields followed by
  StandardScaler.
- `sparse_signed_log` — signed-log transforms with nonzero indicators for
  `X41` and `X23`.

The scaler is fitted on training data only. IV targets remain in decimal units.
The ordinary trainer excludes IV values outside configured bounds `[0, 3]`;
it does not winsorize or median-impute them.

## Project layout

```text
config/default.yaml                Data paths, splits, model and training settings
data/nk-iv-prediction/             Local, Git-ignored competition data
src/baselines/interpolation.py     Interpolation baselines
src/baselines/sabr.py              SABR model implementation
src/data/feature_scaling.py        Configurable market-feature transformations
src/data/preprocess.py             Wide-to-long preparation and preprocessing
src/evaluate/masked_quote_benchmark.py
                                   Masked-quote validation and result plotting
src/evaluate/scaling_comparison.py Feature-scaling diagnostic experiment
src/models/pinn_model.py           PINN architecture and IV physics penalty
src/train/train_and_save.py        Standard feature-to-IV training pipeline
tests/                             Unit tests
results/                           Recorded benchmark CSVs, logs, and plots
```

## Setup and tests

Python 3.9 or newer is required. Install project dependencies with:

```bash
pip install -e .
```

Run the test suite:

```bash
pytest -q
```

## Important limitations and next steps

1. **Validate robustness:** run several mask seeds and multiple chronological
   holdout windows; report mean and variation, not only one score.
2. **Test mask shapes:** compare random quote masks with contiguous strike gaps
   and other realistic missingness patterns.
3. **Strengthen baselines:** review SABR calibration and interpolation
   behavior, including sparse and boundary cases.
4. **Inspect financial quality:** add surface smoothness and static-arbitrage
   diagnostics; prediction error alone does not establish a valid surface.
5. **Add recent, diverse data:** obtain labeled option surfaces from multiple
   asset classes and market regimes. Evaluate both within-asset and held-out
   asset generalization. The current NIFTY 50 dataset cannot answer that
   generalization question.
6. **Resolve strike-grid extrapolation:** the 10 competition test-only strikes
   are outside the training strike coverage and cannot be scored offline with
   the current labels.
7. **Complete production evaluation:** the legacy `pinn-evaluate` and
   `pinn-compare` CLI commands are placeholders; the masked-quote benchmark is
   the current end-to-end comparison path.

## References

- Hagan et al. (2002), “Managing Smile Risk” — SABR implied-volatility
  approximation.
- Raissi et al. (2019), “Physics-informed neural networks: A deep learning
  framework for solving forward and inverse problems.”
- Brooks et al. (2017), “A Topological Approach to Scaling in Financial Data,”
  [arXiv:1710.08860](https://arxiv.org/abs/1710.08860). This analyzes
  self-similarity in ordered price series; it is not a feature-scaling method
  for the anonymous `X` inputs.

## License

The package metadata declares MIT. See the repository license file for the
applicable terms.
