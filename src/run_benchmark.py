"""Этап 3: основной бенчмарк.

Для каждого датасета, seed из SEEDS и модели из MODELS:
  split 80/20 (random_state=seed) -> обучение -> метрики + время.
Каждый прогон сохраняется в results/raw/main/<dataset>__<model>__s<seed>.json,
поэтому скрипт можно перезапускать: готовые прогоны пропускаются.
В конце собираются benchmark_results_per_seed.csv и benchmark_results.csv.

Запуск:  python src/run_benchmark.py [--datasets a,b] [--models X,Y] [--seeds 0,1,2]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASETS, MODELS, RAW, ROOT, SEEDS, load_dataset, load_json, run_model, save_json, split  # noqa: E402

# Порядок — от дешёвых к дорогим (для TabPFN на CPU), чтобы ранние результаты появились быстрее.
ORDER = ["sklearn_diabetes", "pima_diabetes", "cmc", "credit_g", "yeast", "abalone", "ozone_level_8hr",
         "us_crime"]
MAIN_DIR = RAW / "main"


def run_path(ds: str, model: str, seed: int):
    return MAIN_DIR / f"{ds}__{model}__s{seed}.json"


def run(datasets, models, seeds) -> None:
    for ds in datasets:
        X, y, task, _, meta = load_dataset(ds)
        n_classes = len(meta["classes"]) if task != "regression" else None
        for seed in seeds:
            Xtr, Xte, ytr, yte = split(X, y, task, seed)
            for model in models:
                path = run_path(ds, model, seed)
                if path.exists() and load_json(path).get("status") == "ok":
                    continue
                t = time.perf_counter()
                rec = run_model(model, task, Xtr, ytr, Xte, yte, seed,
                                pos_label=meta.get("pos_label"), n_classes=n_classes)
                rec.update(dataset=ds, task=task, experiment="main")
                save_json(path, rec)
                main_metric = (rec.get("metrics") or {}).get("rmse" if task == "regression" else "accuracy")
                print(f"[{time.strftime('%H:%M:%S')}] {ds:16s} s{seed} {model:12s} {rec['status']:5s} "
                      f"metric={main_metric} time={time.perf_counter() - t:.1f}s "
                      f"{rec.get('error', '')[:120]}", flush=True)
        print(f"[{time.strftime('%H:%M:%S')}] DATASET DONE {ds}", flush=True)


def collect() -> pd.DataFrame:
    rows = []
    for p in sorted(MAIN_DIR.glob("*.json")):
        r = load_json(p)
        row = {k: r.get(k) for k in ["dataset", "task", "model", "seed", "status", "n_train", "n_test",
                                     "n_features", "fit_time_s", "predict_time_s", "total_time_s",
                                     "refit_time_s", "cv_best_score", "error"]}
        row.update(r.get("metrics") or {})
        row["best_params"] = r.get("best_params")
        rows.append(row)
    return pd.DataFrame(rows)


METRICS = ["accuracy", "f1", "f1_macro", "roc_auc", "pr_auc", "log_loss", "rmse", "mae", "r2"]
TIMES = ["total_time_s", "fit_time_s", "predict_time_s"]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df["status"] == "ok"]
    out = []
    for (ds, model), g in ok.groupby(["dataset", "model"], sort=False):
        row = {"dataset": ds, "task": g["task"].iloc[0], "model": model, "n_seeds": len(g),
               "n_train": int(g["n_train"].iloc[0]), "n_features": int(g["n_features"].iloc[0])}
        for c in METRICS + TIMES:
            if c in g and g[c].notna().any():
                row[f"{c}_mean"] = float(np.mean(g[c]))
                row[f"{c}_std"] = float(np.std(g[c], ddof=1)) if len(g) > 1 else 0.0
        out.append(row)
    agg = pd.DataFrame(out)
    order = {d: i for i, d in enumerate(ORDER)}
    morder = {m: i for i, m in enumerate(MODELS)}
    agg = agg.sort_values(by=["dataset", "model"], key=lambda s: s.map(order) if s.name == "dataset" else s.map(morder))
    return agg.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(ORDER))
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    if not a.aggregate_only:
        datasets = [d for d in a.datasets.split(",") if d in DATASETS]
        run(datasets, a.models.split(","), [int(s) for s in a.seeds.split(",")])
    df = collect()
    df.to_csv(ROOT / "benchmark_results_per_seed.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(ROOT / "benchmark_results.csv", index=False, float_format="%.6g")
    errors = df[df["status"] != "ok"]
    print(f"runs: {len(df)} ok: {int((df['status'] == 'ok').sum())} errors: {len(errors)}")
    if len(errors):
        print(errors[["dataset", "model", "seed", "error"]].to_string())


if __name__ == "__main__":
    main()
