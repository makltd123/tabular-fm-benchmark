"""Общие компоненты бенчмарка: датасеты, подготовка данных, модели, метрики, замер времени.

Все этапы (run_benchmark.py, run_stress.py) импортируют отсюда, чтобы модели в
основном бенчмарке и в стресс-тестах запускались одинаково.
"""
from __future__ import annotations

import json
import os
import time
import traceback
import warnings
from pathlib import Path

# TabPFN по умолчанию запрещает на CPU выборки > 1000 строк (защита от долгих прогонов).
# GPU на машине нет, поэтому снимаем эту защиту явно; лимиты самой модели (10k строк,
# 500 признаков, 10 классов) при этом продолжают действовать.
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")
# Без телеметрии TabPFN.
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, train_test_split

warnings.filterwarnings("ignore")

# Все модели используют все логические потоки CPU: классика — n_jobs=-1 / thread_count=-1,
# torch (TabPFN, TabICL) — по умолчанию брал бы только физические ядра (2 из 4 потоков);
# замер показал, что 4 потока быстрее на ~25-30% при идентичных предсказаниях.
import torch  # noqa: E402

torch.set_num_threads(os.cpu_count() or 1)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
FIGURES = ROOT / "figures"

SEEDS = [0, 1, 2]
TEST_SIZE = 0.2
N_SEARCH_ITER = 20
CV_FOLDS = 3

MODELS = ["XGBoost", "CatBoost", "RandomForest", "TabPFN-v2", "TabICL-v2"]
CLASSIC = {"XGBoost", "CatBoost", "RandomForest"}
FOUNDATION = {"TabPFN-v2", "TabICL-v2"}

# ---------------------------------------------------------------------------
# Датасеты
# ---------------------------------------------------------------------------
# role — какие категории разнообразия из ТЗ закрывает датасет.
DATASETS: dict[str, dict] = {
    "pima_diabetes": dict(
        source="openml", openml_id=37, task="binary",
        roles=["small", "binary"],
        note="Pima Indians Diabetes: медицинские показатели, предсказание диабета.",
    ),
    "credit_g": dict(
        source="openml", openml_id=31, task="binary",
        roles=["medium", "binary", "categorical-heavy"],
        note="German Credit: кредитный риск, 13 из 20 признаков категориальные.",
    ),
    "ozone_level_8hr": dict(
        source="openml", openml_id=1487, task="binary",
        roles=["medium", "binary", "imbalanced"],
        note="Ozone Level (8-hour peak): ~6.3% положительного класса, 72 признака.",
    ),
    "cmc": dict(
        source="openml", openml_id=23, task="multiclass",
        roles=["medium", "multiclass", "categorical-heavy"],
        note="Contraceptive Method Choice: 3 класса, смешанные категориальные/числовые признаки.",
    ),
    "sklearn_diabetes": dict(
        source="sklearn", loader="load_diabetes", task="regression",
        roles=["small", "regression"],
        note="sklearn diabetes: прогрессирование диабета через год, 442 строки.",
    ),
    "abalone": dict(
        source="openml", openml_id=42726, task="regression",
        roles=["medium", "regression", "categorical"],
        note="Abalone: возраст моллюска (число колец) по физическим измерениям, признак Sex категориальный.",
    ),
    "yeast": dict(
        source="openml", openml_id=181, task="multiclass",
        roles=["medium", "multiclass", "imbalanced"],
        note="Yeast: локализация белка, 10 классов (= лимит TabPFN v2), классы от 463 до 5 объектов.",
    ),
    "us_crime": dict(
        source="openml", openml_id=315, task="regression",
        roles=["medium", "high-dimensional", "regression", "real-missing-values"],
        drop_cols=["state", "county", "community", "communityname", "fold"],
        note="Communities and Crime (UCI): уровень насильственных преступлений по данным переписи; "
             "122 признака, в 23 колонках до 84% пропусков. Служебные колонки (штат, округ, "
             "название общины, номер фолда) удалены как непредсказательные.",
    ),
}

# Отдельный датасет только для зонда «число классов выше лимита» в стресс-тесте
# (ни один из основных датасетов по условию ТЗ не превышает лимит классов).
EXTRA_DATASETS: dict[str, dict] = {
    "letter": dict(
        source="openml", openml_id=6, task="multiclass", roles=["many-classes-probe"],
        note="Letter recognition: 26 классов, 16 признаков. Используется только в стресс-тесте числа классов.",
    ),
}


def load_dataset(name: str):
    """Возвращает (X: DataFrame, y: ndarray, task, cat_cols, meta).

    Категориальные признаки имеют dtype 'category', числовые — float64.
    Для классификации y закодирован в 0..K-1; meta['classes'] хранит исходные метки,
    meta['pos_label'] — индекс миноритарного класса (для бинарных задач).
    """
    spec = {**DATASETS, **EXTRA_DATASETS}[name]
    if spec["source"] == "openml":
        import openml

        ds = openml.datasets.get_dataset(
            spec["openml_id"], download_data=True,
            download_qualities=False, download_features_meta_data=True,
        )
        X, y, cat_ind, _ = ds.get_data(target=ds.default_target_attribute)
        source_url = f"https://www.openml.org/d/{spec['openml_id']}"
        version = ds.version
        licence = ds.licence
    else:
        from sklearn import datasets as skd

        bunch = getattr(skd, spec["loader"])(as_frame=True)
        X, y = bunch.data, bunch.target
        cat_ind = [False] * X.shape[1]
        source_url = f"sklearn.datasets.{spec['loader']}"
        version = None
        licence = "BSD-3-Clause (scikit-learn); исходные данные Efron et al. 2004"

    X = X.copy()
    cat_cols = [c for c, is_cat in zip(X.columns, cat_ind) if is_cat]
    drop = spec.get("drop_cols", [])
    if drop:
        X = X.drop(columns=drop)
        cat_cols = [c for c in cat_cols if c not in drop]
    for c in X.columns:
        if c in cat_cols:
            X[c] = X[c].astype("category")
        else:
            X[c] = X[c].astype("float64")

    meta = dict(source_url=source_url, openml_version=version, licence=licence)
    task = spec["task"]
    if task == "regression":
        y = np.asarray(y, dtype="float64")
    else:
        y_ser = pd.Series(y).astype(str)
        classes = sorted(y_ser.unique())
        mapping = {c: i for i, c in enumerate(classes)}
        y = y_ser.map(mapping).to_numpy()
        counts = np.bincount(y)
        meta["classes"] = classes
        meta["class_counts"] = counts.tolist()
        meta["pos_label"] = int(np.argmin(counts))
    return X, y, task, cat_cols, meta


def split(X, y, task, seed):
    strat = y if task != "regression" else None
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=strat)


# ---------------------------------------------------------------------------
# Представление данных для разных моделей
# ---------------------------------------------------------------------------
def to_ordinal(X: pd.DataFrame) -> np.ndarray:
    """Категории -> целочисленные коды (NaN сохраняется), всё в float64."""
    out = X.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            codes = out[c].cat.codes.astype("float64")
            codes[codes < 0] = np.nan
            out[c] = codes
    return out.to_numpy(dtype="float64")


def to_catboost(X: pd.DataFrame) -> pd.DataFrame:
    """CatBoost требует строки/целые в категориальных колонках (NaN -> отдельная категория)."""
    out = X.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype(object).where(out[c].notna(), "__nan__").astype(str)
    return out


def cat_indices(X: pd.DataFrame) -> list[int]:
    return [i for i, c in enumerate(X.columns) if isinstance(X[c].dtype, pd.CategoricalDtype)]


def simple_impute(Xtr: pd.DataFrame, Xte: pd.DataFrame):
    """«Обычная» импутация для классики в стресс-тесте: медиана для числовых,
    мода для категориальных; статистики считаются только по train."""
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for c in Xtr.columns:
        if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
            mode = Xtr[c].mode(dropna=True)
            fill = mode.iloc[0] if len(mode) else Xtr[c].cat.categories[0]
        else:
            fill = Xtr[c].median()
            if pd.isna(fill):
                fill = 0.0
        Xtr[c] = Xtr[c].fillna(fill)
        Xte[c] = Xte[c].fillna(fill)
    return Xtr, Xte


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------
def xgb_space():
    return {
        "n_estimators": randint(100, 1001),
        "learning_rate": loguniform(0.01, 0.3),
        "max_depth": randint(2, 11),
        "min_child_weight": loguniform(1, 20),
        "subsample": uniform(0.5, 0.5),
        "colsample_bytree": uniform(0.5, 0.5),
        "reg_lambda": loguniform(1e-2, 10),
        "reg_alpha": loguniform(1e-3, 1),
    }


def catboost_space():
    return {
        "iterations": randint(200, 1001),
        "learning_rate": loguniform(0.01, 0.3),
        "depth": randint(4, 9),
        "l2_leaf_reg": loguniform(1, 10),
        "bagging_temperature": uniform(0, 1),
        "random_strength": loguniform(0.1, 10),
    }


def make_classic(model: str, task: str, seed: int, cat_idx: list[int], params: dict | None = None):
    params = dict(params or {})
    clf = task != "regression"
    if model == "XGBoost":
        from xgboost import XGBClassifier, XGBRegressor

        cls = XGBClassifier if clf else XGBRegressor
        return cls(random_state=seed, n_jobs=-1, tree_method="hist", enable_categorical=True,
                   max_cat_to_onehot=1, **params)
    if model == "CatBoost":
        from catboost import CatBoostClassifier, CatBoostRegressor

        cls = CatBoostClassifier if clf else CatBoostRegressor
        # cat_features передаётся в fit() (см. classic_fit_kwargs): в конструкторе CatBoost
        # ломает sklearn.clone внутри RandomizedSearchCV.
        return cls(random_seed=seed, thread_count=-1, verbose=0, allow_writing_files=False, **params)
    if model == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if clf else RandomForestRegressor
        return cls(random_state=seed, n_jobs=-1, **params)
    raise ValueError(model)


def classic_fit_kwargs(model: str, cat_idx: list[int]) -> dict:
    return {"cat_features": cat_idx} if (model == "CatBoost" and cat_idx) else {}


def classic_input(model: str, X: pd.DataFrame):
    if model == "XGBoost":
        return X  # нативная поддержка category dtype
    if model == "CatBoost":
        return to_catboost(X)
    return to_ordinal(X)  # RandomForest: ordinal-коды, NaN поддерживается в sklearn >= 1.4


def make_foundation(model: str, task: str, seed: int, cat_idx: list[int]):
    clf = task != "regression"
    if model == "TabPFN-v2":
        # ВАЖНО: версия задаётся явно. По умолчанию tabpfn>=9 грузит TabPFN-3.5,
        # веса которого под некоммерческой лицензией. Веса v2 — Prior Labs License
        # (Apache 2.0 + атрибуция «Built with PriorLabs-TabPFN»).
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        from tabpfn.constants import ModelVersion
        from tabpfn.model_loading import _load_checkpoint_cached, clear_built_model_cache

        clear_built_model_cache()
        _load_checkpoint_cached.cache_clear()  # загрузка весов входит в замер времени
        cls = TabPFNClassifier if clf else TabPFNRegressor
        return cls.create_default_for_version(
            ModelVersion.V2, random_state=seed, categorical_features_indices=cat_idx or None,
        )
    if model == "TabICL-v2":
        from tabicl import TabICLClassifier, TabICLRegressor

        cls = TabICLClassifier if clf else TabICLRegressor
        # Параметры по умолчанию (чекпоинт *-v2-20260212); задаём только seed.
        return cls(random_state=seed)
    raise ValueError(model)


def foundation_input(model: str, X: pd.DataFrame):
    if model == "TabPFN-v2":
        return to_ordinal(X)  # + categorical_features_indices в конструкторе
    return X  # TabICL сам находит категориальные колонки по dtype и кодирует их


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------
def compute_metrics(task: str, y_true, pred, proba=None, pos_label: int | None = None, n_classes=None):
    m: dict[str, float] = {}
    if task == "regression":
        m["rmse"] = float(root_mean_squared_error(y_true, pred))
        m["mae"] = float(mean_absolute_error(y_true, pred))
        m["r2"] = float(r2_score(y_true, pred))
        return m
    m["accuracy"] = float(accuracy_score(y_true, pred))
    if task == "binary":
        m["f1"] = float(f1_score(y_true, pred, pos_label=pos_label, average="binary"))
        m["f1_macro"] = float(f1_score(y_true, pred, average="macro"))
        if proba is not None:
            p = proba[:, pos_label]
            yb = (np.asarray(y_true) == pos_label).astype(int)
            m["roc_auc"] = float(roc_auc_score(yb, p))
            m["pr_auc"] = float(average_precision_score(yb, p))
            m["log_loss"] = float(log_loss(y_true, np.clip(proba, 1e-15, 1), labels=list(range(proba.shape[1]))))
    else:
        m["f1"] = float(f1_score(y_true, pred, average="macro"))
        m["f1_macro"] = m["f1"]
        if proba is not None:
            labels = list(range(proba.shape[1]))
            pr = np.clip(proba, 1e-15, 1)
            pr = pr / pr.sum(axis=1, keepdims=True)
            try:
                m["roc_auc"] = float(roc_auc_score(y_true, pr, multi_class="ovr", average="macro", labels=labels))
            except ValueError:
                m["roc_auc"] = float("nan")
            m["log_loss"] = float(log_loss(y_true, pr, labels=labels))
    return m


def _full_proba(model, proba, n_classes):
    """Выравнивает столбцы predict_proba по всем классам 0..K-1 (на случай, если в
    обучающей выборке встретились не все классы)."""
    classes = np.asarray(getattr(model, "classes_", np.arange(proba.shape[1]))).astype(int)
    if len(classes) == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return proba
    full = np.zeros((proba.shape[0], n_classes))
    full[:, classes] = proba
    return full


# ---------------------------------------------------------------------------
# Один прогон модели
# ---------------------------------------------------------------------------
def run_model(model: str, task: str, Xtr: pd.DataFrame, ytr, Xte: pd.DataFrame, yte, seed: int,
              pos_label=None, n_classes=None, tune: bool = True, params: dict | None = None,
              impute: bool = False) -> dict:
    """Обучает модель, считает метрики и время.

    Классика: tune=True — RandomizedSearchCV (N_SEARCH_ITER итераций, CV_FOLDS фолдов);
    время = подбор + финальное обучение (refit) + предсказание.
    tune=False — обучение с переданными params (или дефолтами).
    impute=True — перед обучением классики пропуски заполняются simple_impute.
    Foundation: без настройки; время = создание модели (загрузка весов) + fit + predict.
    """
    rec: dict = dict(model=model, seed=seed, n_train=len(ytr), n_test=len(yte), n_features=Xtr.shape[1])
    cat_idx = cat_indices(Xtr)
    clf = task != "regression"
    if clf and n_classes is None:
        n_classes = int(max(np.max(ytr), np.max(yte)) + 1)
    try:
        t0 = time.perf_counter()
        if model in CLASSIC:
            if impute:
                Xtr, Xte = simple_impute(Xtr, Xte)
            Xtr_m, Xte_m = classic_input(model, Xtr), classic_input(model, Xte)
            if tune and model != "RandomForest":
                base = make_classic(model, task, seed, cat_idx)
                space = xgb_space() if model == "XGBoost" else catboost_space()
                cv = (StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed) if clf
                      else KFold(CV_FOLDS, shuffle=True, random_state=seed))
                scoring = "neg_log_loss" if clf else "neg_root_mean_squared_error"
                est = RandomizedSearchCV(base, space, n_iter=N_SEARCH_ITER, cv=cv, scoring=scoring,
                                         random_state=seed, n_jobs=1, refit=True, error_score="raise")
                est.fit(Xtr_m, ytr, **classic_fit_kwargs(model, cat_idx))
                rec["best_params"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in est.best_params_.items()}
                rec["cv_best_score"] = float(est.best_score_)
                rec["refit_time_s"] = float(est.refit_time_)
                fitted = est.best_estimator_
            else:
                fitted = make_classic(model, task, seed, cat_idx, params)
                fitted.fit(Xtr_m, ytr, **classic_fit_kwargs(model, cat_idx))
                if params:
                    rec["best_params"] = params
            t1 = time.perf_counter()
            pred_in = Xte_m
        else:
            fitted = make_foundation(model, task, seed, cat_idx)
            Xtr_m, Xte_m = foundation_input(model, Xtr), foundation_input(model, Xte)
            fitted.fit(Xtr_m, ytr)
            t1 = time.perf_counter()
            pred_in = Xte_m

        if clf:
            proba = _full_proba(fitted, np.asarray(fitted.predict_proba(pred_in)), n_classes)
            pred = proba.argmax(axis=1)
        else:
            proba = None
            pred = np.asarray(fitted.predict(pred_in)).ravel()
        t2 = time.perf_counter()
        rec.update(status="ok", fit_time_s=t1 - t0, predict_time_s=t2 - t1, total_time_s=t2 - t0)
        rec["metrics"] = compute_metrics(task, yte, pred, proba, pos_label, n_classes)
    except Exception as e:  # фиксируем ошибку как результат (например, превышение лимита классов)
        rec.update(status="error", error=f"{type(e).__name__}: {e}"[:2000],
                   traceback=traceback.format_exc()[-3000:])
    return rec


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))
