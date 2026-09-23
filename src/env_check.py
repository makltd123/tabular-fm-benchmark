"""Этап 1: проверка окружения.

- импортирует все библиотеки и печатает версии;
- проверяет GPU (torch.cuda.is_available());
- заранее скачивает веса TabPFN v2 и TabICL v2 (чтобы скачивание не попало в замеры времени);
- пишет results/environment.json и requirements.txt (полный pip freeze с точными версиями).
"""
from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys

sys.path.insert(0, os.path.dirname(__file__))
from common import RESULTS, ROOT, save_json  # noqa: E402

LIBS = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "catboost", "torch",
        "tabpfn", "tabicl", "openml", "matplotlib", "huggingface_hub"]


def main() -> None:
    info: dict = {"python": sys.version, "platform": platform.platform(),
                  "processor": platform.processor(), "cpu_count_logical": os.cpu_count()}
    from importlib.metadata import version as dist_version

    versions = {}
    for lib in LIBS:
        importlib.import_module(lib)
        versions[lib] = dist_version("scikit-learn" if lib == "sklearn" else lib.replace("_", "-"))
        print(f"import {lib:16s} OK  {versions[lib]}")
    info["versions"] = versions

    import psutil
    import torch

    info["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    info["cpu_count_physical"] = psutil.cpu_count(logical=False)
    info["torch_threads"] = torch.get_num_threads()
    info["cuda_available"] = bool(torch.cuda.is_available())
    info["device"] = "cuda" if info["cuda_available"] else "cpu"
    print("CUDA available:", info["cuda_available"], "| torch threads:", info["torch_threads"])

    # Предзагрузка весов (одноразовое скачивание, не входит в бенчмарк).
    from tabicl import TabICLClassifier, TabICLRegressor
    from tabpfn import TabPFNClassifier, TabPFNRegressor
    from tabpfn.constants import ModelVersion
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    yc = (X[:, 0] > 0).astype(int)
    yr = X[:, 0] * 2.0
    checkpoints = {}
    for name, m, y in [
        ("TabPFN-v2 classifier", TabPFNClassifier.create_default_for_version(ModelVersion.V2, n_estimators=1), yc),
        ("TabPFN-v2 regressor", TabPFNRegressor.create_default_for_version(ModelVersion.V2, n_estimators=1), yr),
        ("TabICL-v2 classifier", TabICLClassifier(n_estimators=1), yc),
        ("TabICL-v2 regressor", TabICLRegressor(n_estimators=1), yr),
    ]:
        m.fit(X, y)
        m.predict(X[:5])
        ckpt = getattr(m, "model_path", None) or getattr(m, "checkpoint_version", None)
        checkpoints[name] = str(ckpt)
        print(f"{name}: weights OK ({ckpt})")
    info["checkpoints"] = checkpoints
    save_json(RESULTS / "environment.json", info)

    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    header = (
        "# Полный список пакетов окружения бенчмарка (pip freeze), Python "
        f"{platform.python_version()}, {platform.system()} {platform.release()}.\n"
        "# torch собран для CPU, ставится с индекса PyTorch:\n"
        "--extra-index-url https://download.pytorch.org/whl/cpu\n"
    )
    (ROOT / "requirements.txt").write_text(header + freeze, encoding="utf-8")
    print("requirements.txt written:", len(freeze.splitlines()), "packages")


if __name__ == "__main__":
    main()
