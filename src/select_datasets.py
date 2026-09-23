"""Этап 2: загрузка выбранных датасетов и запись datasets_manifest.json."""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASETS, EXTRA_DATASETS, ROOT, TEST_SIZE, load_dataset, save_json  # noqa: E402

# Лимиты моделей, сверенные с исходным кодом установленных версий (см. AGENT_LOG.md).
LIMITS = {
    "TabPFN-v2": {"max_samples": 10_000, "max_features": 500, "max_classes": 10},
    "TabICL-v2": {"recommended_train_samples": "300-48000", "recommended_features": "2-100",
                  "max_classes_native": 10, "many_classes": "support_many_classes=True (иерархическая схема)"},
}


def describe(name: str, spec: dict, main: bool) -> dict:
    X, y, task, cat_cols, meta = load_dataset(name)
    n = len(y)
    d = dict(
        name=name, in_main_benchmark=main, source=spec["source"], url=meta["source_url"],
        openml_id=spec.get("openml_id"), openml_version=meta.get("openml_version"), licence=meta["licence"],
        task=task, n_rows=int(n), n_features=int(X.shape[1]), n_categorical=len(cat_cols),
        categorical_columns=[str(c) for c in cat_cols], n_missing_cells=int(X.isna().sum().sum()),
        n_train=int(n - math.ceil(n * TEST_SIZE)),  # как в train_test_split
        roles=spec["roles"], description=spec["note"],
    )
    if task != "regression":
        counts = np.bincount(y)
        d.update(n_classes=int(len(counts)), class_counts=counts.tolist(),
                 minority_share=round(float(counts.min() / n), 4), original_labels=meta["classes"],
                 minority_label=meta["classes"][meta["pos_label"]])
    else:
        d.update(target_mean=float(np.mean(y)), target_std=float(np.std(y)))
    within = (d["n_train"] <= 10_000 and d["n_features"] <= 500 and d.get("n_classes", 0) <= 10)
    d["within_tabpfn_v2_limits"] = bool(within)
    return d


def main() -> None:
    items = [describe(n, s, True) for n, s in DATASETS.items()]
    items += [describe(n, s, False) for n, s in EXTRA_DATASETS.items()]
    coverage = {}
    for role in ["small", "medium", "high-dimensional", "imbalanced", "binary", "multiclass", "regression",
                 "categorical-heavy", "real-missing-values"]:
        coverage[role] = [d["name"] for d in items if d["in_main_benchmark"] and role in d["roles"]]
    manifest = dict(
        description="Датасеты бенчмарка TabPFN v2 / TabICL v2 vs XGBoost / CatBoost / RandomForest",
        split=f"случайный train/test {int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}, стратифицированный для классификации; "
              "временной компоненты в выбранных данных нет",
        model_limits=LIMITS, coverage_of_tz_categories=coverage, datasets=items,
    )
    save_json(ROOT / "datasets_manifest.json", manifest)
    for d in items:
        print(f"{d['name']:18s} {d['task']:10s} rows={d['n_rows']:6d} feats={d['n_features']:4d} "
              f"cat={d['n_categorical']:2d} classes={d.get('n_classes', '-')} minority={d.get('minority_share', '-')}")
    print("coverage:", coverage)


if __name__ == "__main__":
    main()
