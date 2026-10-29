"""Fixtures compartilhadas pelos testes.

- Gera dados sintéticos com o mesmo esquema do dataset real, para que os testes
  rodem rápido e não dependam do CSV de verdade.
- Isola o MLflow em um banco SQLite próprio por teste (nada vai para mlflow_data/).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.pipeline import Pipeline

from projetofase2.config import MlflowSettings, Params, load_params, load_settings
from projetofase2.data import load_raw_data, split_features_target
from projetofase2.model import build_pipeline

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Dados sintéticos
# --------------------------------------------------------------------------- #
def make_shoppers_df(n: int = 600, seed: int = 0, n_duplicates: int = 30) -> pd.DataFrame:
    """Gera um DataFrame com o mesmo esquema do dataset real.

    - ~20% de compras, com sinal forte em PageValues (como no dataset real);
    - uma categoria rara em TrafficType (valor 20, uma única linha);
    - `n_duplicates` linhas duplicadas ao final.
    """
    rng = np.random.default_rng(seed)
    page_values = np.where(rng.random(n) < 0.75, 0.0, rng.exponential(15, n))
    revenue = rng.random(n) < (0.04 + 0.5 * (page_values > 0))

    df = pd.DataFrame({
        "Administrative": rng.poisson(2.3, n),
        "Administrative_Duration": rng.exponential(80, n),
        "Informational": rng.poisson(0.5, n),
        "Informational_Duration": rng.exponential(30, n),
        "ProductRelated": rng.poisson(30, n),
        "ProductRelated_Duration": rng.exponential(1200, n),
        "BounceRates": rng.exponential(0.02, n),
        "ExitRates": rng.exponential(0.04, n),
        "PageValues": page_values,
        "SpecialDay": rng.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], n,
                                 p=[0.9, 0.02, 0.02, 0.02, 0.02, 0.02]),
        "Month": rng.choice(MONTHS, n),
        "OperatingSystems": rng.integers(1, 9, n),
        "Browser": rng.integers(1, 14, n),
        "Region": rng.integers(1, 10, n),
        "TrafficType": rng.integers(1, 6, n),
        "VisitorType": rng.choice(["Returning_Visitor", "New_Visitor", "Other"], n,
                                  p=[0.85, 0.14, 0.01]),
        "Weekend": rng.random(n) < 0.23,
        "Revenue": revenue,
    })
    df.loc[0, "TrafficType"] = 20  # categoria rara

    if n_duplicates:
        df = pd.concat([df, df.sample(n_duplicates, random_state=seed)], ignore_index=True)
    return df


def write_params_file(tmp_path: Path, raw_csv: Path) -> Path:
    """Arquivo de parâmetros com uma busca minúscula (2 combinações, 3 folds)."""
    content = {
        "random_state": RANDOM_STATE,
        "data": {
            "raw_path": str(raw_csv),
            "processed_dir": str(tmp_path / "processed"),
            "test_size": 0.2,
        },
        "train": {
            "cv_splits": 3,
            "n_iter": 2,
            "n_jobs": 1,
            "scoring": "average_precision",
            "param_distributions": {"rf__n_estimators": [10, 20], "rf__max_depth": [3, 5]},
            "model_path": str(tmp_path / "models" / "model.joblib"),
            "metrics_path": str(tmp_path / "reports" / "metrics.json"),
        },
    }
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(content), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def shoppers_df() -> pd.DataFrame:
    return make_shoppers_df()


@pytest.fixture
def shoppers_csv(tmp_path, shoppers_df) -> Path:
    path = tmp_path / "online_shoppers_intention.csv"
    shoppers_df.to_csv(path, index=False)
    return path


@pytest.fixture
def clean_df(shoppers_csv) -> pd.DataFrame:
    """DataFrame já passado por `load_raw_data` (sem duplicatas, tipos corretos)."""
    return load_raw_data(shoppers_csv)


@pytest.fixture
def xy(clean_df) -> tuple[pd.DataFrame, pd.Series]:
    return split_features_target(clean_df)


@pytest.fixture
def small_pipeline() -> Pipeline:
    """Pipeline real, mas com poucas árvores para os testes serem rápidos."""
    return build_pipeline(RANDOM_STATE).set_params(rf__n_estimators=25)


@pytest.fixture
def params_file(tmp_path, shoppers_csv) -> Path:
    return write_params_file(tmp_path, shoppers_csv)


@pytest.fixture
def params(params_file) -> Params:
    return load_params(params_file)


@pytest.fixture(autouse=True)
def isolated_mlflow(monkeypatch, tmp_path) -> str:
    """Cada teste usa um banco SQLite próprio em tmp_path. As variáveis de ambiente
    têm prioridade sobre o .env do projeto, então nada vai para mlflow_data/."""
    uri = f"sqlite:///{(tmp_path / 'mlflow_store' / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "exp-teste")
    monkeypatch.setenv("MLFLOW_MODEL_NAME", "modelo_teste")
    monkeypatch.setenv("MLFLOW_REGISTER_MODEL", "true")
    monkeypatch.setenv("MLFLOW_CHAMPION_ALIAS", "champion")
    monkeypatch.setenv("MLFLOW_SELECTION_METRIC", "test_pr_auc")
    return uri


@pytest.fixture
def settings(isolated_mlflow) -> MlflowSettings:
    return load_settings(env_path=None)
