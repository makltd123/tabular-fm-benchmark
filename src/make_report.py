"""Этап 5 (техническая часть): таблицы и графики для FINAL_REPORT.md.

Читает benchmark_results_per_seed.csv и results/stress_results_per_seed.csv,
пишет:
  results/report_tables.md      — markdown-таблицы (вставляются в FINAL_REPORT.md)
  results/stress_results.csv    — агрегированные (mean ± std) результаты стресс-тестов
  results/deltas.csv            — разница «лучшая foundation vs лучшая классика» по датасетам
  figures/*.png                 — графики
Текст выводов пишется вручную по этим цифрам.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import DATASETS, EXTRA_DATASETS, FIGURES, MODELS, RESULTS, ROOT  # noqa: E402
from run_benchmark import ORDER  # noqa: E402

# Палитра: слоты 1–5 эталонной категориальной палитры (проверена валидатором:
# CVD ΔE ≥ 9.1, normal-vision ΔE ≥ 19.6). Цвет закреплён за моделью, не за рангом.
COLORS = {"TabPFN-v2": "#2a78d6", "TabICL-v2": "#eb6834", "XGBoost": "#1baf7a",
          "CatBoost": "#eda100", "RandomForest": "#e87ba4"}
MARKERS = {"TabPFN-v2": "o", "TabICL-v2": "s", "XGBoost": "^", "CatBoost": "D", "RandomForest": "v"}
PLOT_ORDER = ["TabPFN-v2", "TabICL-v2", "XGBoost", "CatBoost", "RandomForest"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

TASKS = {name: spec["task"] for name, spec in {**DATASETS, **EXTRA_DATASETS}.items()}
PRIMARY = {"binary": ("roc_auc", "ROC-AUC", True), "multiclass": ("accuracy", "Accuracy", True),
           "regression": ("rmse", "RMSE", False)}
TITLES = {"sklearn_diabetes": "sklearn diabetes (регрессия)", "pima_diabetes": "Pima diabetes (бинарная)",
          "cmc": "CMC (3 класса)", "credit_g": "German credit (бинарная)", "abalone": "Abalone (регрессия)",
          "ozone_level_8hr": "Ozone 8hr (дисбаланс 6%)", "yeast": "Yeast (10 классов)",
          "us_crime": "US crime (122 призн., регрессия)", "letter": "letter"}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fmt(m, s, digits=3):
    if pd.isna(m):
        return "—"
    return f"{m:.{digits}f} ± {s:.{digits}f}" if not pd.isna(s) else f"{m:.{digits}f}"


def fmt_time(x):
    if pd.isna(x):
        return "—"
    return f"{x:.1f}" if x < 100 else f"{x:.0f}"


# ---------------------------------------------------------------------------
# Основной бенчмарк
# ---------------------------------------------------------------------------
def main_tables(per_seed: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    ok = per_seed[per_seed.status == "ok"]
    g = ok.groupby(["dataset", "model"])
    mean, std = g.mean(numeric_only=True), g.std(numeric_only=True)
    lines = []

    # 1. Сводная таблица по основной метрике
    lines.append("### Сводная таблица (основная метрика, среднее ± std по 3 seed)\n")
    lines.append("Основная метрика: ROC-AUC для бинарной классификации, accuracy для многоклассовой, "
                 "RMSE для регрессии (меньше — лучше). **Жирным** — лучший результат в строке.\n")
    lines.append("| Датасет | Метрика | " + " | ".join(MODELS) + " |")
    lines.append("|---|---|" + "---|" * len(MODELS))
    for ds in ORDER:
        task = ok[ok.dataset == ds].task.iloc[0]
        key, label, higher = PRIMARY[task]
        vals = {m: (mean.loc[(ds, m), key], std.loc[(ds, m), key]) if (ds, m) in mean.index else (np.nan, np.nan)
                for m in MODELS}
        best = (max if higher else min)([v[0] for v in vals.values() if not pd.isna(v[0])])
        cells = []
        for m in MODELS:
            c = fmt(*vals[m])
            cells.append(f"**{c}**" if vals[m][0] == best else c)
        lines.append(f"| {ds} | {label} | " + " | ".join(cells) + " |")

    # 2. Полные таблицы по задачам
    blocks = [("binary", ["accuracy", "f1", "roc_auc", "pr_auc", "log_loss"]),
              ("multiclass", ["accuracy", "f1_macro", "roc_auc", "log_loss"]),
              ("regression", ["rmse", "mae", "r2"])]
    names = {"accuracy": "Accuracy", "f1": "F1 (миноритарный класс)", "f1_macro": "F1-macro", "roc_auc": "ROC-AUC",
             "pr_auc": "PR-AUC", "log_loss": "LogLoss", "rmse": "RMSE", "mae": "MAE", "r2": "R²"}
    for task, cols in blocks:
        dss = [d for d in ORDER if (ok[ok.dataset == d].task == task).any()]
        lines.append(f"\n### Все метрики: {task}\n")
        lines.append("| Датасет | Модель | " + " | ".join(names[c] for c in cols) + " | Время, с |")
        lines.append("|---|---|" + "---|" * (len(cols) + 1))
        for ds in dss:
            for m in MODELS:
                if (ds, m) not in mean.index:
                    lines.append(f"| {ds} | {m} | " + " | ".join("—" for _ in cols) + " | — |")
                    continue
                r, s = mean.loc[(ds, m)], std.loc[(ds, m)]
                lines.append(f"| {ds} | {m} | " + " | ".join(fmt(r[c], s[c]) for c in cols)
                             + f" | {fmt_time(r['total_time_s'])} |")

    # 3. Время
    lines.append("\n### Время «от данных до предсказаний», секунды (среднее по 3 seed)\n")
    lines.append("Классика (XGBoost, CatBoost): RandomizedSearchCV (20 итераций × 3 фолда) + финальное обучение + "
                 "предсказание. RandomForest: обучение с параметрами по умолчанию + предсказание. "
                 "Foundation-модели: загрузка весов + fit + предсказание, без настройки. CPU, без GPU.\n")
    lines.append("| Датасет | n_train × признаков | " + " | ".join(MODELS) + " |")
    lines.append("|---|---|" + "---|" * len(MODELS))
    for ds in ORDER:
        sub = ok[ok.dataset == ds]
        shape = f"{int(sub.n_train.iloc[0])} × {int(sub.n_features.iloc[0])}"
        cells = [fmt_time(mean.loc[(ds, m), "total_time_s"]) if (ds, m) in mean.index else "—" for m in MODELS]
        lines.append(f"| {ds} | {shape} | " + " | ".join(cells) + " |")
    tot = ok.groupby("model")["total_time_s"].sum() / ok.seed.nunique()
    lines.append(f"| **Сумма по {len(ORDER)} датасетам** | | " + " | ".join(fmt_time(tot.get(m, np.nan)) for m in MODELS) + " |")

    # 4. Разница foundation vs классика
    rows = []
    for ds in ORDER:
        task = ok[ok.dataset == ds].task.iloc[0]
        metrics = ["accuracy", "f1", "roc_auc"] if task != "regression" else ["rmse", "mae", "r2"]
        for met in metrics:
            higher = met not in ("rmse", "mae")
            v = {m: mean.loc[(ds, m), met] for m in MODELS if (ds, m) in mean.index}
            fm = {m: v[m] for m in ["TabPFN-v2", "TabICL-v2"] if m in v}
            cm = {m: v[m] for m in ["XGBoost", "CatBoost", "RandomForest"] if m in v}
            pick = max if higher else min
            bf = pick(fm, key=fm.get)
            bc = pick(cm, key=cm.get)
            row = dict(dataset=ds, task=task, metric=met, best_foundation=bf, best_foundation_value=fm[bf],
                       best_classic=bc, best_classic_value=cm[bc])
            if met in ("rmse", "mae"):
                row["delta"] = (cm[bc] - fm[bf]) / cm[bc] * 100  # % снижения ошибки (+ = foundation лучше)
                row["delta_unit"] = "% ошибки"
            elif met == "r2":
                row["delta"] = (fm[bf] - cm[bc])
                row["delta_unit"] = "R² (абс.)"
            else:
                row["delta"] = (fm[bf] - cm[bc]) * 100
                row["delta_unit"] = "п.п."
            # отдельные сравнения для каждой foundation-модели с лучшей классикой
            for f in ["TabPFN-v2", "TabICL-v2"]:
                if f in v:
                    if met in ("rmse", "mae"):
                        row[f"{f}_vs_best_classic"] = (cm[bc] - v[f]) / cm[bc] * 100
                    elif met == "r2":
                        row[f"{f}_vs_best_classic"] = v[f] - cm[bc]
                    else:
                        row[f"{f}_vs_best_classic"] = (v[f] - cm[bc]) * 100
            rows.append(row)
    deltas = pd.DataFrame(rows)
    lines.append("\n### Насколько foundation-модели лучше/хуже лучшей классической модели\n")
    lines.append("Для accuracy/F1/ROC-AUC — разница в процентных пунктах (п.п.); для RMSE/MAE — "
                 "относительное снижение ошибки в %; для R² — абсолютная разница. "
                 "Положительное значение = foundation-модель лучше. «Лучшая классика» выбирается "
                 "по каждой метрике отдельно (среди XGBoost/CatBoost/RandomForest).\n")
    lines.append("| Датасет | Метрика | Лучшая классика | TabPFN-v2 vs она | TabICL-v2 vs она |")
    lines.append("|---|---|---|---|---|")
    for _, r in deltas.iterrows():
        unit = r["delta_unit"]

        def sg(x, unit=unit):
            if pd.isna(x):
                return "—"
            return f"{x:+.2f} {unit}" if unit != "R² (абс.)" else f"{x:+.3f}"
        lines.append(f"| {r['dataset']} | {r['metric']} | {r['best_classic']} ({r['best_classic_value']:.3f}) | "
                     f"{sg(r.get('TabPFN-v2_vs_best_classic'))} | {sg(r.get('TabICL-v2_vs_best_classic'))} |")

    # 5. Средний ранг
    ranks = []
    for ds in ORDER:
        task = ok[ok.dataset == ds].task.iloc[0]
        key, _, higher = PRIMARY[task]
        s = pd.Series({m: mean.loc[(ds, m), key] for m in MODELS if (ds, m) in mean.index})
        ranks.append(s.rank(ascending=not higher))
    rk = pd.concat(ranks, axis=1).mean(axis=1).sort_values()
    lines.append(f"\n### Средний ранг по основной метрике (1 = лучший, по {len(ORDER)} датасетам)\n")
    lines.append("| Модель | Средний ранг |")
    lines.append("|---|---|")
    for m, v in rk.items():
        lines.append(f"| {m} | {v:.2f} |")

    # 6. Попарное сравнение на одинаковых разбиениях (датасет × seed)
    lines.append("\n### Попарное сравнение на одинаковых разбиениях (8 датасетов × 3 seed = 24 пары)\n")
    lines.append("Модели сравниваются на одном и том же train/test (тот же seed), по основной метрике. "
                 "«Побед» — в скольких парах foundation-модель лучше; «средняя разница» — среднее по парам "
                 "(п.п. для ROC-AUC/accuracy; % снижения RMSE для регрессии; + = foundation лучше).\n")
    lines.append("| Foundation | vs | Побед | Средняя разница | Мин. / макс. разница |")
    lines.append("|---|---|---|---|---|")
    for f in ["TabPFN-v2", "TabICL-v2"]:
        for c in ["XGBoost", "CatBoost", "RandomForest"]:
            diffs = []
            for ds in ORDER:
                task = ok[ok.dataset == ds].task.iloc[0]
                key, _, higher = PRIMARY[task]
                sub = ok[ok.dataset == ds].pivot_table(index="seed", columns="model", values=key)
                if f not in sub or c not in sub:
                    continue
                if higher:
                    d = (sub[f] - sub[c]) * 100
                else:
                    d = (sub[c] - sub[f]) / sub[c] * 100
                diffs += list(d.dropna())
            diffs = np.array(diffs)
            wins = int((diffs > 0).sum())
            lines.append(f"| {f} | {c} | {wins} из {len(diffs)} | {diffs.mean():+.2f} | "
                         f"{diffs.min():+.2f} / {diffs.max():+.2f} |")
    return "\n".join(lines), deltas


def fig_main(per_seed: pd.DataFrame):
    ok = per_seed[per_seed.status == "ok"]
    fig, axes = plt.subplots(3, 3, figsize=(15, 9.2), facecolor=SURFACE)
    axes = axes.ravel()
    for ax, ds in zip(axes, ORDER):
        style_axes(ax)
        sub = ok[ok.dataset == ds]
        key, label, higher = PRIMARY[sub.task.iloc[0]]
        for i, m in enumerate(PLOT_ORDER):
            v = sub[sub.model == m][key]
            if not len(v):
                continue
            ax.errorbar(v.mean(), i, xerr=v.std(ddof=1), fmt=MARKERS[m], color=COLORS[m], ms=8,
                        mec=SURFACE, mew=1.5, ecolor=COLORS[m], elinewidth=2, capsize=0)
        ax.set_yticks(range(len(PLOT_ORDER)), PLOT_ORDER)
        ax.invert_yaxis()
        ax.set_title(TITLES[ds], fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel(label + (" (↑ лучше)" if higher else " (↓ лучше)"), fontsize=8, color=INK2)
        ax.grid(axis="y", visible=False)
    ax = axes[-1]
    ax.axis("off")
    ax.text(0, 0.9, "Точка — среднее по 3 seed,\nлиния — ± 1 std.\n\nROC-AUC — бинарные задачи,\n"
            "Accuracy — многоклассовые,\nRMSE — регрессия.", fontsize=9, color=INK2, va="top")
    fig.suptitle(f"Качество моделей на {len(ORDER)} датасетах (тестовая выборка 20%)", fontsize=12, color=INK, x=0.01,
                 ha="left")
    fig.tight_layout()
    fig.savefig(FIGURES / "main_quality.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_time(per_seed: pd.DataFrame):
    ok = per_seed[per_seed.status == "ok"]
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    style_axes(ax)
    n = len(PLOT_ORDER)
    width = 0.8 / n
    for i, m in enumerate(PLOT_ORDER):
        vals = [ok[(ok.dataset == ds) & (ok.model == m)].total_time_s.mean() for ds in ORDER]
        x = np.arange(len(ORDER)) + (i - (n - 1) / 2) * width
        ax.bar(x, vals, width=width * 0.85, color=COLORS[m], label=m)
    ax.set_yscale("log")
    ax.set_xticks(range(len(ORDER)), [TITLES[d].split(" (")[0] for d in ORDER], fontsize=8)
    ax.set_ylabel("секунды, лог. шкала", fontsize=8, color=INK2)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=8, ncol=5, loc="upper left", bbox_to_anchor=(0, 1.12))
    ax.set_title("Время от данных до предсказаний (подбор + обучение + inference), CPU", fontsize=11,
                 color=INK, loc="left", pad=28)
    fig.tight_layout()
    fig.savefig(FIGURES / "main_time.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Стресс-тесты
# ---------------------------------------------------------------------------
def stress_frame(per_seed: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    """Добавляет к стресс-результатам «нулевой» уровень из основного бенчмарка."""
    ok_main = per_seed[per_seed.status == "ok"].copy()
    parts = [stress]
    # missing: уровень 0 = основной бенчмарк; «native NaN»-варианты на уровне 0 совпадают с базовыми
    base = ok_main[ok_main.dataset.isin(stress[stress.experiment == "missing"].dataset.unique())].copy()
    base["experiment"], base["level"] = "missing", 0.0
    nat = base[base.model.isin(["XGBoost", "CatBoost", "RandomForest"])].copy()
    nat["model"] = nat["model"] + " (native NaN)"
    parts += [base, nat]
    # tiny_train: полный train из основного бенчмарка
    tiny = ok_main[ok_main.dataset.isin(stress[stress.experiment == "tiny_train"].dataset.unique())].copy()
    tiny["experiment"] = "tiny_train"
    tiny["level"] = tiny["n_train"].astype(float)
    parts.append(tiny)
    # n_classes: k=10 = основной бенчмарк yeast
    k10 = ok_main[ok_main.dataset == "yeast"].copy()
    k10["experiment"], k10["level"] = "n_classes", 10.0
    parts.append(k10)
    df = pd.concat(parts, ignore_index=True)
    return df


def agg_stress(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "f1", "roc_auc", "pr_auc", "rmse", "mae", "r2", "total_time_s"]
    ok = df[df.status == "ok"]
    g = ok.groupby(["experiment", "dataset", "model", "level"])
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_ok"] = g.size()
    err = df[df.status == "error"].groupby(["experiment", "dataset", "model", "level"]).size().rename("n_error")
    out = out.join(err, how="outer").reset_index()
    out["n_error"] = out["n_error"].fillna(0).astype(int)
    out["n_ok"] = out["n_ok"].fillna(0).astype(int)
    return out


def primary_for(ds, task):
    return PRIMARY["binary" if task == "binary" else task]


def fig_stress_lines(agg: pd.DataFrame, exp: str, datasets: list[str], xlabel: str, fname: str, title: str,
                     logx=False, include_native=False, metric_override=None, limit_line=None):
    from matplotlib.lines import Line2D

    single = len(datasets) == 1
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.4 if single else 4.6 * len(datasets), 4.6 if single else 3.9),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    models = PLOT_ORDER + ([m + " (native NaN)" for m in ["XGBoost", "CatBoost", "RandomForest"]]
                           if include_native else [])
    for ax, ds in zip(axes, datasets):
        style_axes(ax)
        sub = agg[(agg.experiment == exp) & (agg.dataset == ds)]
        key, label, higher = metric_override or PRIMARY[TASKS[ds]]
        for m in models:
            s = sub[sub.model == m].sort_values("level")
            s = s[s[f"{key}_mean"].notna()]
            if not len(s):
                continue
            base = m.replace(" (native NaN)", "")
            native = m != base
            ax.errorbar(s.level, s[f"{key}_mean"], yerr=s[f"{key}_std"], color=COLORS[base], lw=2 if not native else 1.5,
                        ls="--" if native else "-", marker=MARKERS[base], ms=6 if not native else 5,
                        mfc=COLORS[base] if not native else SURFACE, mec=COLORS[base] if native else SURFACE,
                        mew=1.5, capsize=0, elinewidth=0.8, alpha=0.95)
        if limit_line is not None:
            ax.axvline(limit_line, color=MUTED, lw=1)
            y0, y1 = ax.get_ylim()
            ax.set_ylim(y0, y1 + 0.12 * (y1 - y0))  # место под подпись над линиями
            ax.text(limit_line + 0.4, y1 + 0.1 * (y1 - y0),
                    "лимит TabPFN v2 (10 классов):\nвыше — ошибка валидации", fontsize=7.5, color=INK2,
                    va="top")
        if logx:
            ax.set_xscale("log")
            lv = sorted(sub.level.dropna().unique())
            ax.set_xticks(lv, [str(int(x)) for x in lv])
            ax.minorticks_off()
        ax.set_title(TITLES.get(ds, ds), fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel(xlabel, fontsize=8, color=INK2)
        ax.set_ylabel(label + (" (↑ лучше)" if higher else " (↓ лучше)"), fontsize=8, color=INK2)
    # Легенда из «чистых» образцов линий: у errorbar-хэндлов усы выглядят как пунктир.
    handles, labels = [], []
    for m in models:
        base = m.replace(" (native NaN)", "")
        native = m != base
        handles.append(Line2D([], [], color=COLORS[base], lw=2 if not native else 1.5, ls="--" if native else "-",
                              marker=MARKERS[base], ms=6 if not native else 5,
                              mfc=COLORS[base] if not native else SURFACE, mec=COLORS[base] if native else SURFACE,
                              mew=1.5))
        labels.append(m if not native else f"{base}, без импутации (встроенная обработка NaN)")
    ncol = 3 if (include_native or len(datasets) == 1) else 5
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=ncol, loc="upper left", bbox_to_anchor=(0.01, 0.93),
               handlelength=3.6)
    fig.suptitle(title, fontsize=11.5, color=INK, x=0.01, ha="left", y=0.99)
    nrow = int(np.ceil(len(labels) / ncol))
    fig.tight_layout(rect=(0, 0, 1, 0.9 - 0.06 * (nrow - 1)))
    fig.savefig(FIGURES / fname, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def stress_tables(agg: pd.DataFrame) -> str:
    lines = []
    specs = [
        ("missing", "Пропуски (MCAR), доля ячеек", lambda x: f"{int(round(x * 100))}%"),
        ("tiny_train", "Размер обучающей выборки", lambda x: str(int(x))),
        ("n_classes", "Число классов (yeast, k самых частых классов)", lambda x: str(int(x))),
        ("letter_probe", "Число классов (letter, 100 строк/класс)", lambda x: str(int(x))),
    ]
    for exp, title, lf in specs:
        sub = agg[agg.experiment == exp]
        if not len(sub):
            continue
        lines.append(f"\n### {title}\n")
        for ds in sub.dataset.unique():
            s = sub[sub.dataset == ds]
            if s["rmse_mean"].notna().any():
                key, label = "rmse", "RMSE"
            elif exp in ("missing", "tiny_train") and s["roc_auc_mean"].notna().any() and ds == "pima_diabetes":
                key, label = "roc_auc", "ROC-AUC"
            else:
                key, label = "accuracy", "Accuracy"
            levels = sorted(s.level.unique())
            models = [m for m in PLOT_ORDER + [x + " (native NaN)" for x in ["XGBoost", "CatBoost", "RandomForest"]]
                      if m in set(s.model)]
            lines.append(f"**{ds}** — {label} (среднее ± std по seed)\n")
            lines.append("| Модель | " + " | ".join(lf(x) for x in levels) + " |")
            lines.append("|---|" + "---|" * len(levels))
            for m in models:
                cells = []
                for lv in levels:
                    r = s[(s.model == m) & (s.level == lv)]
                    if not len(r):
                        cells.append("—")
                    elif r.n_ok.iloc[0] == 0:
                        cells.append("ошибка")
                    else:
                        cells.append(fmt(r[f"{key}_mean"].iloc[0], r[f"{key}_std"].iloc[0]))
                lines.append(f"| {m} | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    per_seed = pd.read_csv(ROOT / "benchmark_results_per_seed.csv")
    text, deltas = main_tables(per_seed)
    deltas.to_csv(RESULTS / "deltas.csv", index=False, float_format="%.5g")
    fig_main(per_seed)
    fig_time(per_seed)
    sp = RESULTS / "stress_results_per_seed.csv"
    if sp.exists():
        stress = pd.read_csv(sp)
        full = stress_frame(per_seed, stress)
        agg = agg_stress(full)
        agg.to_csv(RESULTS / "stress_results.csv", index=False, float_format="%.5g")
        text += "\n\n## Стресс-тесты\n" + stress_tables(agg)
        sds = [d for d in ["pima_diabetes", "cmc", "abalone"] if d in set(agg.dataset)]
        if (agg.experiment == "missing").any():
            fig_stress_lines(agg, "missing", sds, "доля пропущенных ячеек (train и test)", "stress_missing.png",
                             "Деградация при пропусках (MCAR). Классика: сплошная линия — импутация медианой/модой, "
                             "пунктир — встроенная обработка NaN", include_native=True)
        if (agg.experiment == "tiny_train").any():
            fig_stress_lines(agg, "tiny_train", sds, "строк в обучающей выборке (лог. шкала)", "stress_tiny.png",
                             "Урезанная обучающая выборка (test тот же)", logx=True)
        if (agg.experiment == "n_classes").any():
            fig_stress_lines(agg, "n_classes", ["yeast"], "число классов", "stress_n_classes.png",
                             "Рост числа классов до лимита TabPFN v2 (10)",
                             metric_override=("accuracy", "Accuracy", True))
        if (agg.experiment == "letter_probe").any():
            fig_stress_lines(agg, "letter_probe", ["letter"], "число классов", "stress_letter.png",
                             "Выше лимита: letter, 100 строк на класс",
                             metric_override=("accuracy", "Accuracy", True), limit_line=10)
    (RESULTS / "report_tables.md").write_text(text, encoding="utf-8")
    print("written:", RESULTS / "report_tables.md")


if __name__ == "__main__":
    main()
