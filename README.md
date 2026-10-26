# Tabular foundation models vs. gradient boosting: a reproducible CPU benchmark

[Русская версия](README.ru.md)

**Built with PriorLabs-TabPFN.**

A reproducible benchmark of two tabular foundation models, **TabPFN v2** and **TabICL v2**,
used as-is with no tuning. They are compared against **XGBoost** and **CatBoost** with
hyperparameter search and **RandomForest** with default settings. The benchmark uses
8 public datasets, 3 seeds, 120 main runs and 501 stress-test runs, and everything ran on a
**CPU-only laptop** (Intel i3-1115G4, 8 GB RAM, no GPU).

The full report (in Russian) is in [FINAL_REPORT.md](FINAL_REPORT.md). The agent's decision
log is in [AGENT_LOG.md](AGENT_LOG.md).

## Key results

| Model | Mean rank (1 = best, 8 datasets) | Wins vs. XGBoost (24 dataset×seed pairs) | Wins vs. CatBoost | Time, all 8 datasets, one seed |
|---|---|---|---|---|
| **TabICL v2** (no tuning) | **1.31** | 22 / 24 | 21 / 24 | **291 s** |
| **TabPFN v2** (no tuning) | 2.06 | 21 / 24 | 18 / 24 | 1206 s |
| CatBoost (20×3-fold search) | 3.38 | — | — | 3383 s |
| XGBoost (20×3-fold search) | 3.88 | — | — | 470 s |
| RandomForest (defaults) | 4.38 | — | — | 4 s |

- **Quality.** Without any tuning, both foundation models match or beat the tuned boosters.
  The margin is small: on average +2.3…+2.6 percentage points (or % lower RMSE) for TabICL v2,
  which is comparable to the seed-to-seed spread on a single dataset. The advantage is
  consistent across datasets rather than large on any one of them.
- **Largest gains** show up in regression (3–5% lower RMSE), on minority-class F1 at the
  default threshold, and on very small training sets. With 50 training rows on abalone the
  error is 8.5–9.7% lower than the best classic model.
- **No gain** under 30% missing values (MCAR) on 2 of 3 datasets, and on the categorical-heavy
  `cmc` dataset with truncated training sets. **TabPFN v2 fails hard above 10 classes.**
  TabICL v2 handles 26 classes and stays ahead.
- **Time on CPU.** TabICL v2 is faster than hyperparameter search. TabPFN v2 scales poorly on
  CPU: 7–9 minutes per run on ~2k rows × 72–122 features.

**Scope:** datasets have 442–4177 rows and up to 122 features. The conclusions are not
validated on large tables (10k+ rows), where gradient boosting is traditionally strong.
See section 6 of the report for all limitations.

## Datasets

| Dataset | Source | Task | Rows | Features |
|---|---|---|---|---|
| sklearn_diabetes | `sklearn.datasets` | regression | 442 | 10 |
| pima_diabetes | OpenML 37 | binary | 768 | 8 |
| credit_g | OpenML 31 | binary | 1000 | 20 (13 categorical) |
| cmc | OpenML 23 | 3 classes | 1473 | 9 (7 categorical) |
| yeast | OpenML 181 | 10 classes | 1484 | 8 |
| abalone | OpenML 42726 | regression | 4177 | 8 (1 categorical) |
| ozone_level_8hr | OpenML 1487 | binary, 6.3% positives | 2534 | 72 |
| us_crime | OpenML 315 | regression, real missing values | 1994 | 122 |
| letter (stress test only) | OpenML 6 | 26 classes | 20000 | 16 |

## Reproduce

Requires Python 3.14. The pinned environment is in `requirements.txt`, with a CPU build of torch.

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python src\env_check.py        # stage 1: environment check, pre-download weights
.venv\Scripts\python src\select_datasets.py  # stage 2: datasets_manifest.json
.venv\Scripts\python src\run_benchmark.py    # stage 3: benchmark_results.csv (~4 h on this CPU)
.venv\Scripts\python src\run_stress.py       # stage 4: stress tests (~4 h)
.venv\Scripts\python src\make_report.py      # tables and figures
```

Every run is saved as a JSON file under `results/raw/`. The scripts can be interrupted and
restarted, and finished runs are skipped.

## Repository layout

| Path | Contents |
|---|---|
| `src/common.py` | datasets, model wrappers, metrics, timing (shared by all stages) |
| `src/run_benchmark.py`, `src/run_stress.py`, `src/make_report.py` | stages 3, 4, 5 |
| `benchmark_results.csv`, `benchmark_results_per_seed.csv` | main results: mean ± std / every run with tuned hyperparameters |
| `results/stress_results*.csv` | stress-test results |
| `results/raw/` | one JSON per run |
| `figures/` | charts used in the report |
| `TZ_agent_tabular_benchmark.md` | the original task specification (in Russian) |

## Licenses and attribution

The code and documents in this repository are released under the MIT License, © 2026 Fedor
Makovskiy (see [LICENSE](LICENSE)).

- **TabPFN v2.** Weights `Prior-Labs/TabPFN-v2-clf` and `-reg` are under the Prior Labs
  License 1.1 (Apache 2.0 plus attribution). **Built with PriorLabs-TabPFN.** Reference:
  Hollmann et al., *Accurate predictions on small data with a tabular foundation model*,
  Nature 637, 319–326 (2025). Package `tabpfn` 9.0.0 is Apache 2.0.
- **TabICL v2.** Package `tabicl` 2.2.0 and the `jingang/TabICL` checkpoints are BSD 3-Clause,
  © 2025 Soda team @ Inria. Reference: Qu et al., *TabICL: A Tabular Foundation Model for
  In-Context Learning on Large Data*, ICML 2025.
- XGBoost and CatBoost are Apache 2.0; scikit-learn is BSD 3-Clause.
- **Deliberately not used**, because of non-commercial licenses: TabFM (Google) and TabPFN
  2.5 / 2.6 / 3 / 3.5. Note that `tabpfn` 9.x loads TabPFN-3.5 by default, so the code pins
  v2 explicitly with `create_default_for_version(ModelVersion.V2)`.

---

Author: [Fedor Makovskiy](https://fedor.alqira.ru), AI developer. Services and projects with quality metrics: [fedor.alqira.ru](https://fedor.alqira.ru).
