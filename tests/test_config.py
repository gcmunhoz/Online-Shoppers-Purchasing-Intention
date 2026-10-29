"""Testes das configurações: esquema de colunas, params.yaml e variáveis de ambiente."""
from pathlib import Path

import pytest

from projetofase2 import config
from projetofase2.config import (
    BINARY_COLS,
    CATEGORICAL_COLS,
    DEFAULT_PARAMS_PATH,
    FEATURE_COLS,
    NUMERIC_COLS,
    PROJECT_ROOT,
    TARGET,
    load_params,
    load_settings,
    resolve_path,
)

MLFLOW_VARS = ["MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_NAME", "MLFLOW_MODEL_NAME",
               "MLFLOW_REGISTER_MODEL", "MLFLOW_CHAMPION_ALIAS", "MLFLOW_SELECTION_METRIC"]


# --------------------------------------------------------------------------- #
# Esquema
# --------------------------------------------------------------------------- #
def test_column_groups_are_disjoint_and_exclude_target():
    groups = [set(NUMERIC_COLS), set(CATEGORICAL_COLS), set(BINARY_COLS)]

    assert sum(len(g) for g in groups) == len(set().union(*groups)), "colunas repetidas"
    assert TARGET not in FEATURE_COLS


def test_column_groups_cover_17_features():
    assert len(FEATURE_COLS) == 17  # o dataset tem 17 features + o alvo


def test_column_groups_match_dataset_schema(shoppers_df):
    assert set(FEATURE_COLS + [TARGET]) == set(shoppers_df.columns)


# --------------------------------------------------------------------------- #
# Caminhos e params.yaml
# --------------------------------------------------------------------------- #
def test_project_root_contains_pyproject():
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_resolve_path_is_relative_to_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert resolve_path("data/raw") == PROJECT_ROOT / "data" / "raw"
    assert resolve_path(tmp_path) == tmp_path


def test_project_params_file_loads():
    params = load_params(DEFAULT_PARAMS_PATH)

    assert params.data.raw_path == PROJECT_ROOT / "data" / "raw" / "online_shoppers_intention.csv"
    assert params.data.train_path.parent == params.data.processed_dir
    assert 0 < params.data.test_size < 1
    assert params.train.model_path.is_absolute()
    assert all(k.startswith("rf__") for k in params.train.param_distributions)


def test_load_params_reads_custom_file(params, tmp_path):
    assert params.train.n_iter == 2
    assert params.train.cv_splits == 3
    assert params.data.processed_dir == tmp_path / "processed"


# --------------------------------------------------------------------------- #
# Variáveis de ambiente / .env
# --------------------------------------------------------------------------- #
@pytest.fixture
def clean_env(monkeypatch):
    for var in MLFLOW_VARS:
        monkeypatch.delenv(var, raising=False)


def test_settings_defaults_without_env(clean_env, tmp_path):
    settings = load_settings(env_path=tmp_path / "nao_existe.env")

    assert settings.tracking_uri == config.DEFAULT_TRACKING_URI
    assert Path(settings.tracking_uri.removeprefix("sqlite:///")) == \
        PROJECT_ROOT / "mlflow_data" / "mlflow.db"
    assert settings.register_model is True
    assert settings.champion_alias == "champion"
    assert settings.selection_metric == "test_pr_auc"


def test_settings_read_env_file(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("MLFLOW_EXPERIMENT_NAME=do-arquivo\nMLFLOW_REGISTER_MODEL=false\n",
                        encoding="utf-8")
    settings = load_settings(env_file)

    assert settings.experiment_name == "do-arquivo"
    assert settings.register_model is False


def test_environment_variables_override_env_file(clean_env, tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MLFLOW_EXPERIMENT_NAME=do-arquivo\n", encoding="utf-8")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "do-ambiente")

    assert load_settings(env_file).experiment_name == "do-ambiente"
