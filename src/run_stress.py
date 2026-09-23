"""Этап 4: стресс-тест ограничений.

Эксперименты (каждый прогон -> results/raw/stress/<exp>/*.json, можно перезапускать):

  missing     — MCAR-пропуски 5/15/30% ячеек признаков (train и test) на STRESS_DATASETS.
                Классика: «обычная» импутация (медиана/мода) + гиперпараметры, найденные
                на этапе 3 для того же seed. Дополнительно — те же модели со встроенной
                обработкой NaN (без импутации). Foundation-модели получают NaN как есть.
                Уровень 0% берётся из результатов этапа 3.
  tiny_train  — обучающая выборка урезана до 50/100/250 строк (стратифицированная подвыборка
                из train того же seed), test тот же. Классика заново подбирает гиперпараметры
                (RandomizedSearchCV, 20 итераций) на урезанной выборке.
  n_classes   — yeast (10 классов): k самых частых классов, k = 2, 4, 6, 8 (k = 10 — этап 3).
                Классика — с гиперпараметрами этапа 3 для того же seed.
  letter_probe — дополнительный датасет letter (26 классов): первые k классов, k = 5..26,
                по 100 строк на класс. Проверка поведения выше лимита 10 классов.
                Классика — с параметрами библиотек по умолчанию (без подбора).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (FOUNDATION, RAW, ROOT, SEEDS, load_dataset, load_json, run_model,  # noqa: E402
                    save_json, split)
from run_benchmark import run_path  # noqa: E402

STRESS_DATASETS = ["pima_diabetes", "cmc", "abalone"]
MISSING_RATES = [0.05, 0.15, 0.30]
TINY_SIZES = [50, 100, 250]
YEAST_K = [2, 4, 6, 8]
LETTER_K = [5, 10, 11, 15, 20, 26]
LETTER_PER_CLASS = 100
STRESS_DIR = RAW / "stress"

CLASSIC_BASE = ["XGBoost", "CatBoost", "RandomForest"]
NATIVE_SUFFIX = " (native NaN)"


def main_params(ds: str, model: str, seed: int) -> dict | None:
    if model == "RandomForest":
        return None
    return load_json(run_path(ds, model, seed)).get("best_params")


def _do(path, fn, label):
    if path.exists() and load_json(path).get("status") in ("ok", "error"):
        return
    t = time.perf_counter()
    rec = fn()
    save_json(path, rec)
    m = rec.get("metrics") or {}
    key = "rmse" if "rmse" in m else "accuracy"
    print(f"[{time.strftime('%H:%M:%S')}] {label:60s} {rec['status']:5s} {key}={m.get(key)} "
          f"time={time.perf_counter() - t:.1f}s {rec.get('error', '')[:150]}", flush=True)


def inject_mcar(X: pd.DataFrame, rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(10_000 * seed + int(round(rate * 1000)))
    mask = rng.random(X.shape) < rate
    Xm = X.copy()
    for j, c in enumerate(X.columns):
        Xm.loc[mask[:, j], c] = np.nan
    return Xm


def exp_missing(seeds, models_filter=None):
    for ds in STRESS_DATASETS:
        X, y, task, _, meta = load_dataset(ds)
        ncls = len(meta["classes"]) if task != "regression" else None
        for rate in MISSING_RATES:
            for seed in seeds:
                Xm = inject_mcar(X, rate, seed)
                Xtr, Xte, ytr, yte = split(Xm, y, task, seed)
                jobs = []
                for m in CLASSIC_BASE:
                    p = main_params(ds, m, seed)
                    jobs.append((m, dict(tune=False, params=p, impute=True)))
                    jobs.append((m + NATIVE_SUFFIX, dict(tune=False, params=p, impute=False)))
                jobs += [(m, {}) for m in ["TabPFN-v2", "TabICL-v2"]]
                for label, kw in jobs:
                    if models_filter and label.split(" ")[0] not in models_filter:
                        continue
                    base = label.replace(NATIVE_SUFFIX, "")
                    path = STRESS_DIR / "missing" / f"{ds}__{label.replace(' ', '_')}__r{int(rate*100)}__s{seed}.json"

                    def fn(base=base, kw=kw, label=label):
                        rec = run_model(base, task, Xtr, ytr, Xte, yte, seed,
                                        pos_label=meta.get("pos_label"), n_classes=ncls, **kw)
                        rec.update(dataset=ds, task=task, experiment="missing", model=label, level=rate,
                                   missing_share_train=float(Xtr.isna().to_numpy().mean()))
                        return rec
                    _do(path, fn, f"missing {ds} r={rate} s{seed} {label}")


def exp_tiny(seeds, models_filter=None):
    for ds in STRESS_DATASETS:
        X, y, task, _, meta = load_dataset(ds)
        ncls = len(meta["classes"]) if task != "regression" else None
        for seed in seeds:
            Xtr_full, Xte, ytr_full, yte = split(X, y, task, seed)
            for n in TINY_SIZES:
                from sklearn.model_selection import train_test_split

                strat = ytr_full if task != "regression" else None
                Xtr, _, ytr, _ = train_test_split(Xtr_full, ytr_full, train_size=n, random_state=seed, stratify=strat)
                for model in CLASSIC_BASE + ["TabPFN-v2", "TabICL-v2"]:
                    if models_filter and model not in models_filter:
                        continue
                    path = STRESS_DIR / "tiny_train" / f"{ds}__{model}__n{n}__s{seed}.json"

                    def fn(model=model):
                        rec = run_model(model, task, Xtr, ytr, Xte, yte, seed,
                                        pos_label=meta.get("pos_label"), n_classes=ncls, tune=True)
                        rec.update(dataset=ds, task=task, experiment="tiny_train", level=n)
                        return rec
                    _do(path, fn, f"tiny {ds} n={n} s{seed} {model}")


def exp_n_classes(seeds, models_filter=None):
    ds = "yeast"
    X, y, task, _, meta = load_dataset(ds)
    by_freq = np.argsort(-np.bincount(y), kind="stable")  # классы от самого частого к самому редкому
    for seed in seeds:
        for k in YEAST_K:
            keep = np.sort(by_freq[:k])
            idx = np.isin(y, keep)
            Xk = X[idx].reset_index(drop=True)
            yk = np.searchsorted(keep, y[idx])
            Xtr, Xte, ytr, yte = split(Xk, yk, "multiclass", seed)
            t = "binary" if k == 2 else "multiclass"
            pos = int(np.argmin(np.bincount(yk))) if k == 2 else None
            for model in CLASSIC_BASE + ["TabPFN-v2", "TabICL-v2"]:
                if models_filter and model not in models_filter:
                    continue
                path = STRESS_DIR / "n_classes" / f"{ds}__{model}__k{k}__s{seed}.json"

                def fn(model=model):
                    kw = {} if model in FOUNDATION else dict(tune=False, params=main_params(ds, model, seed))
                    rec = run_model(model, t, Xtr, ytr, Xte, yte, seed, pos_label=pos, n_classes=k, **kw)
                    rec.update(dataset=ds, task=t, experiment="n_classes", level=k,
                               classes=[meta["classes"][i] for i in keep])
                    return rec
                _do(path, fn, f"n_classes {ds} k={k} s{seed} {model}")


def exp_letter(seeds, models_filter=None):
    ds = "letter"
    X, y, task, _, meta = load_dataset(ds)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for k in LETTER_K:
            rows = np.concatenate([rng.choice(np.flatnonzero(y == c), LETTER_PER_CLASS, replace=False)
                                   for c in range(k)])
            Xk = X.iloc[rows].reset_index(drop=True)
            yk = y[rows]
            Xtr, Xte, ytr, yte = split(Xk, yk, "multiclass", seed)
            for model in CLASSIC_BASE + ["TabPFN-v2", "TabICL-v2"]:
                if models_filter and model not in models_filter:
                    continue
                path = STRESS_DIR / "letter_probe" / f"{ds}__{model}__k{k}__s{seed}.json"

                def fn(model=model):
                    kw = {} if model in FOUNDATION else dict(tune=False, params=None)
                    rec = run_model(model, "multiclass", Xtr, ytr, Xte, yte, seed, n_classes=k, **kw)
                    rec.update(dataset=ds, task="multiclass", experiment="letter_probe", level=k)
                    return rec
                _do(path, fn, f"letter k={k} s{seed} {model}")


def collect() -> pd.DataFrame:
    rows = []
    for p in sorted(STRESS_DIR.rglob("*.json")):
        r = load_json(p)
        row = {k: r.get(k) for k in ["experiment", "dataset", "task", "model", "level", "seed", "status",
                                     "n_train", "n_test", "n_features", "total_time_s", "missing_share_train",
                                     "error"]}
        row.update(r.get("metrics") or {})
        rows.append(row)
    return pd.DataFrame(rows)


EXPS = {"missing": exp_missing, "tiny_train": exp_tiny, "n_classes": exp_n_classes, "letter_probe": exp_letter}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", default=",".join(EXPS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--models", default="")
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    mf = [m for m in a.models.split(",") if m] or None
    if not a.collect_only:
        for e in a.exps.split(","):
            EXPS[e](seeds, mf)
    df = collect()
    df.to_csv(ROOT / "results" / "stress_results_per_seed.csv", index=False)
    print("stress runs:", len(df), "ok:", int((df["status"] == "ok").sum()) if len(df) else 0)


if __name__ == "__main__":
    main()
