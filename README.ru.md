# Табличные foundation-модели против градиентного бустинга: воспроизводимый бенчмарк на CPU

[English version](README.md)

**Built with PriorLabs-TabPFN.**

Воспроизводимый бенчмарк двух табличных foundation-моделей — **TabPFN v2** и **TabICL v2** —
без какой-либо настройки. Они сравниваются с **XGBoost** и **CatBoost** с подбором
гиперпараметров и с **RandomForest** с параметрами по умолчанию. В бенчмарке 8 открытых
датасетов, 3 seed, 120 прогонов в основной части и 501 в стресс-тестах. Всё посчитано на
**ноутбуке без GPU** (Intel i3-1115G4, 8 ГБ RAM).

Полный отчёт — [FINAL_REPORT.md](FINAL_REPORT.md), журнал решений —
[AGENT_LOG.md](AGENT_LOG.md).

## Главные результаты

| Модель | Средний ранг (1 — лучший, 8 датасетов) | Побед над XGBoost (24 пары датасет×seed) | Побед над CatBoost | Время на 8 датасетов, один seed |
|---|---|---|---|---|
| **TabICL v2** (без настройки) | **1.31** | 22 / 24 | 21 / 24 | **291 с** |
| **TabPFN v2** (без настройки) | 2.06 | 21 / 24 | 18 / 24 | 1206 с |
| CatBoost (подбор 20×3 фолда) | 3.38 | — | — | 3383 с |
| XGBoost (подбор 20×3 фолда) | 3.88 | — | — | 470 с |
| RandomForest (по умолчанию) | 4.38 | — | — | 4 с |

- **Качество.** Без настройки обе foundation-модели не хуже настроенных бустингов, а чаще
  немного лучше. Отрыв небольшой: у TabICL v2 в среднем +2.3…+2.6 п.п. (или % снижения
  RMSE). Это сопоставимо с разбросом между seed на отдельном датасете: преимущество устойчивое
  по многим датасетам, но не большое ни на одном из них.
- **Сильнее всего выигрывают** в регрессии (RMSE ниже на 3–5%), на F1 редких классов при
  пороге по умолчанию и на очень маленьких обучающих выборках. На abalone при 50 строках
  ошибка на 8.5–9.7% ниже лучшей классики.
- **Не выигрывают** при 30% пропусков (MCAR) на 2 из 3 датасетов и на cmc (много
  категориальных признаков) при урезанной выборке. **TabPFN v2 падает с ошибкой при числе
  классов больше 10.** TabICL v2 работает до 26 классов и остаётся впереди.
- **Время на CPU.** TabICL v2 быстрее подбора гиперпараметров. TabPFN v2 на CPU плохо
  масштабируется: 7–9 минут на прогон при ~2 тыс. строк × 72–122 признака.

**Границы применимости:** в датасетах от 442 до 4177 строк и до 122 признаков. На больших
таблицах (10 тыс.+ строк), где бустинги традиционно сильны, выводы не проверялись. Все
ограничения — в разделе 6 отчёта.

## Датасеты

| Датасет | Источник | Задача | Строк | Признаков |
|---|---|---|---|---|
| sklearn_diabetes | `sklearn.datasets` | регрессия | 442 | 10 |
| pima_diabetes | OpenML 37 | бинарная | 768 | 8 |
| credit_g | OpenML 31 | бинарная | 1000 | 20 (13 категориальных) |
| cmc | OpenML 23 | 3 класса | 1473 | 9 (7 категориальных) |
| yeast | OpenML 181 | 10 классов | 1484 | 8 |
| abalone | OpenML 42726 | регрессия | 4177 | 8 (1 категориальный) |
| ozone_level_8hr | OpenML 1487 | бинарная, 6.3% позитивов | 2534 | 72 |
| us_crime | OpenML 315 | регрессия, реальные пропуски | 1994 | 122 |
| letter (только стресс-тест) | OpenML 6 | 26 классов | 20000 | 16 |

## Как воспроизвести

Нужен Python 3.14. Точные версии пакетов — в `requirements.txt`, torch в CPU-сборке.
Виртуальное окружение лучше создавать вне OneDrive: torch весит несколько ГБ.

```powershell
py -3.14 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python src\env_check.py        # этап 1: проверка окружения, скачивание весов
.venv\Scripts\python src\select_datasets.py  # этап 2: datasets_manifest.json
.venv\Scripts\python src\run_benchmark.py    # этап 3: benchmark_results.csv (~4 ч на этом CPU)
.venv\Scripts\python src\run_stress.py       # этап 4: стресс-тесты (~4 ч)
.venv\Scripts\python src\make_report.py      # таблицы и графики
```

Каждый прогон сохраняется отдельным JSON в `results/raw/`. Скрипты можно прерывать и
перезапускать, готовые прогоны пропускаются.

## Структура репозитория

| Путь | Что внутри |
|---|---|
| `src/common.py` | датасеты, обёртки моделей, метрики, замер времени (общие для всех этапов) |
| `src/run_benchmark.py`, `src/run_stress.py`, `src/make_report.py` | этапы 3, 4, 5 |
| `benchmark_results.csv`, `benchmark_results_per_seed.csv` | основные результаты: среднее ± std / каждый прогон с подобранными гиперпараметрами |
| `results/stress_results*.csv` | результаты стресс-тестов |
| `results/raw/` | по одному JSON на прогон |
| `figures/` | графики из отчёта |
| `TZ_agent_tabular_benchmark.md` | исходное техническое задание |

## Лицензии и атрибуция

Код и документы в этом репозитории распространяются по лицензии MIT, © 2026 Fedor Makovskiy
(см. [LICENSE](LICENSE)).

- **TabPFN v2.** Веса `Prior-Labs/TabPFN-v2-clf` и `-reg` — под Prior Labs License 1.1
  (Apache 2.0 + условие атрибуции). **Built with PriorLabs-TabPFN.** Ссылка: Hollmann et al.,
  *Accurate predictions on small data with a tabular foundation model*, Nature 637, 319–326
  (2025). Пакет `tabpfn` 9.0.0 — Apache 2.0.
- **TabICL v2.** Пакет `tabicl` 2.2.0 и чекпоинты `jingang/TabICL` — BSD 3-Clause, © 2025
  Soda team @ Inria. Ссылка: Qu et al., *TabICL: A Tabular Foundation Model for In-Context
  Learning on Large Data*, ICML 2025.
- XGBoost и CatBoost — Apache 2.0; scikit-learn — BSD 3-Clause.
- **Намеренно не использовались** из-за некоммерческих лицензий: TabFM (Google) и TabPFN
  2.5 / 2.6 / 3 / 3.5. Пакет `tabpfn` 9.x по умолчанию загружает TabPFN-3.5, поэтому в коде
  версия v2 закреплена явно: `create_default_for_version(ModelVersion.V2)`.

---

Автор: [Федор Маковский](https://fedor.alqira.ru), ИИ-разработчик. Услуги и проекты с цифрами качества: [fedor.alqira.ru](https://fedor.alqira.ru).
