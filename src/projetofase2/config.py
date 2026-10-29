"""Configurações do projeto: caminhos, esquema do dataset, parâmetros e ambiente.

- Parâmetros de modelagem (versionados pelo DVC) ficam em ``configs/params.yaml``.
- Configurações de ambiente (MLflow) vêm de variáveis de ambiente ou do ``.env``.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import dotenv_values

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PARAMS_PATH = PROJECT_ROOT / "configs" / "params.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"
DEFAULT_TRACKING_URI = f"sqlite:///{(PROJECT_ROOT / 'mlflow_data' / 'mlflow.db').as_posix()}"

# Esquema do dataset Online Shoppers Purchasing Intention
TARGET = "Revenue"
NUMERIC_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]
CATEGORICAL_COLS = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType",
]
BINARY_COLS = ["Weekend"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS


def resolve_path(path: str | Path) -> Path:
    """Resolve caminhos relativos a partir da raiz do projeto (e não do cwd)."""
    path = Path(path)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


# =======================================
# Parâmetros (configs/params.yaml)
# =======================================

@dataclass(frozen=True)
class DataParams:
    """Origem dos dados brutos e como dividi-los em treino/teste."""

    raw_path: Path
    processed_dir: Path
    test_size: float

    @property
    def train_path(self) -> Path:
        return self.processed_dir / "train.csv"

    @property
    def test_path(self) -> Path:
        return self.processed_dir / "test.csv"


@dataclass(frozen=True)
class TrainParams:
    """Busca de hiperparâmetros e destino dos artefatos do treino."""

    cv_splits: int
    n_iter: int
    n_jobs: int
    scoring: str
    param_distributions: dict[str, list[Any]]
    model_path: Path
    metrics_path: Path


@dataclass(frozen=True)
class Params:
    random_state: int
    data: DataParams
    train: TrainParams


def load_params(path: str | Path = DEFAULT_PARAMS_PATH) -> Params:
    """Lê o arquivo YAML de parâmetros e devolve uma estrutura tipada."""
    with open(path, encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    data, train = raw["data"], raw["train"]
    return Params(
        random_state=int(raw["random_state"]),
        data=DataParams(
            raw_path=resolve_path(data["raw_path"]),
            processed_dir=resolve_path(data["processed_dir"]),
            test_size=float(data["test_size"]),
        ),
        train=TrainParams(
            cv_splits=int(train["cv_splits"]),
            n_iter=int(train["n_iter"]),
            n_jobs=int(train["n_jobs"]),
            scoring=str(train["scoring"]),
            param_distributions=dict(train["param_distributions"]),
            model_path=resolve_path(train["model_path"]),
            metrics_path=resolve_path(train["metrics_path"]),
        ),
    )


# =======================================
# Ambiente (.env / variáveis de ambiente)
# =======================================

@dataclass(frozen=True)
class MlflowSettings:
    """Para onde e como o MLflow registra as runs e o modelo."""

    tracking_uri: str
    experiment_name: str
    model_name: str
    register_model: bool
    champion_alias: str
    selection_metric: str


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "sim"}


def load_settings(env_path: str | Path | None = DEFAULT_ENV_PATH) -> MlflowSettings:
    """Monta as configurações do MLflow.

    Prioridade: variável de ambiente > arquivo ``.env`` > valor padrão.
    O ``.env`` é opcional e não altera ``os.environ``.
    """
    file_values = dotenv_values(env_path) if env_path and Path(env_path).exists() else {}
    values = {**file_values, **os.environ}

    def get(key: str, default: str) -> str:
        return values.get(key) or default

    return MlflowSettings(
        tracking_uri=get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
        experiment_name=get("MLFLOW_EXPERIMENT_NAME", "online-shoppers-purchase-intention"),
        model_name=get("MLFLOW_MODEL_NAME", "rf_online_shoppers"),
        register_model=_parse_bool(get("MLFLOW_REGISTER_MODEL", "true")),
        champion_alias=get("MLFLOW_CHAMPION_ALIAS", "champion"),
        selection_metric=get("MLFLOW_SELECTION_METRIC", "test_pr_auc"),
    )
