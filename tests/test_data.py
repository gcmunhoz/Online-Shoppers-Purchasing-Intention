"""Testes da carga de dados, das constantes de colunas e dos caminhos."""
import importlib
from pathlib import Path

import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
# load_data
# --------------------------------------------------------------------------- #
def test_load_data_removes_duplicates(pipeline_module, shoppers_df, shoppers_csv, capsys):
    expected = len(shoppers_df.drop_duplicates())
    df = pipeline_module.load_data(str(shoppers_csv))

    assert len(df) == expected
    assert len(df) < len(shoppers_df)
    assert not df.duplicated().any()
    assert "->" in capsys.readouterr().out  # log de antes -> depois


def test_load_data_resets_index(pipeline_module, shoppers_csv):
    df = pipeline_module.load_data(str(shoppers_csv))
    assert list(df.index) == list(range(len(df)))


def test_load_data_target_and_weekend_are_int(pipeline_module, shoppers_csv):
    df = pipeline_module.load_data(str(shoppers_csv))

    assert pd.api.types.is_integer_dtype(df[pipeline_module.TARGET])
    assert set(df[pipeline_module.TARGET].unique()) <= {0, 1}
    assert pd.api.types.is_integer_dtype(df["Weekend"])
    assert set(df["Weekend"].unique()) <= {0, 1}


def test_load_data_categoricals_are_strings(pipeline_module, shoppers_csv):
    df = pipeline_module.load_data(str(shoppers_csv))

    for col in pipeline_module.CATEGORICAL_COLS:
        assert all(isinstance(v, str) for v in df[col]), f"{col} deveria conter apenas str"
    # IDs numéricos viram texto, não números ordenados
    assert "1" in set(df["Region"])


def test_load_data_keeps_numeric_values(pipeline_module, shoppers_df, shoppers_csv):
    df = pipeline_module.load_data(str(shoppers_csv))
    expected = shoppers_df.drop_duplicates().reset_index(drop=True)

    for col in pipeline_module.NUMERIC_COLS:
        pd.testing.assert_series_equal(df[col], expected[col], check_dtype=False,
                                       check_exact=False, rtol=1e-9)


def test_load_data_missing_target_raises(pipeline_module, shoppers_df, tmp_path):
    path = tmp_path / "sem_alvo.csv"
    shoppers_df.drop(columns=["Revenue"]).to_csv(path, index=False)
    with pytest.raises(KeyError):
        pipeline_module.load_data(str(path))


def test_load_data_missing_file_raises(pipeline_module, tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline_module.load_data(str(tmp_path / "nao_existe.csv"))


# --------------------------------------------------------------------------- #
# Constantes de colunas
# --------------------------------------------------------------------------- #
def test_column_groups_are_disjoint_and_exclude_target(pipeline_module):
    groups = [set(pipeline_module.NUMERIC_COLS),
              set(pipeline_module.CATEGORICAL_COLS),
              set(pipeline_module.BINARY_COLS)]

    assert sum(len(g) for g in groups) == len(set().union(*groups)), "colunas repetidas entre grupos"
    assert pipeline_module.TARGET not in set().union(*groups)


def test_column_groups_cover_17_features(pipeline_module):
    n = (len(pipeline_module.NUMERIC_COLS) + len(pipeline_module.CATEGORICAL_COLS)
         + len(pipeline_module.BINARY_COLS))
    assert n == 17  # o dataset tem 17 features + o alvo


def test_column_groups_exist_in_schema(pipeline_module, shoppers_df):
    cols = (pipeline_module.NUMERIC_COLS + pipeline_module.CATEGORICAL_COLS
            + pipeline_module.BINARY_COLS + [pipeline_module.TARGET])
    assert set(cols) == set(shoppers_df.columns)


# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
def test_default_paths_are_absolute_and_named_correctly(pipeline_module):
    dataset = Path(pipeline_module.CAMINHO_DATASET)
    model = Path(pipeline_module.CAMINHO_MODELO)

    assert dataset.is_absolute() and model.is_absolute()
    assert dataset.name == "online_shoppers_intention.csv"
    assert dataset.parent.name == "datasets"
    assert model.name == "rf_online_shoppers.joblib"
    assert model.parent.name == "models"


def test_default_paths_point_to_project_root_folders(pipeline_module):
    """src/projetofase2/model.py -> ../../datasets e ../../models = pastas da raiz."""
    root = Path(__file__).resolve().parent.parent

    assert Path(pipeline_module.CAMINHO_DATASET) == root / "datasets" / "online_shoppers_intention.csv"
    assert Path(pipeline_module.CAMINHO_MODELO) == root / "models" / "rf_online_shoppers.joblib"


def test_default_paths_do_not_depend_on_cwd(pipeline_module, tmp_path, monkeypatch):
    before = (pipeline_module.CAMINHO_DATASET, pipeline_module.CAMINHO_MODELO)
    monkeypatch.chdir(tmp_path)
    reloaded = importlib.reload(pipeline_module)

    assert (reloaded.CAMINHO_DATASET, reloaded.CAMINHO_MODELO) == before


# --------------------------------------------------------------------------- #
# Dataset real (pulado automaticamente se o CSV não estiver no caminho)
# --------------------------------------------------------------------------- #
@pytest.fixture
def real_csv(pipeline_module):
    path = Path(pipeline_module.CAMINHO_DATASET)
    if not path.exists():
        pytest.skip(f"Dataset real não encontrado em {path}")
    return path


def test_real_dataset_schema(pipeline_module, real_csv):
    raw = pd.read_csv(real_csv)
    expected = (pipeline_module.NUMERIC_COLS + pipeline_module.CATEGORICAL_COLS
                + pipeline_module.BINARY_COLS + [pipeline_module.TARGET])

    assert set(expected) <= set(raw.columns)
    assert len(raw) > 10_000
    assert raw[expected].isna().sum().sum() == 0


def test_real_dataset_after_load(pipeline_module, real_csv):
    df = pipeline_module.load_data(str(real_csv))
    rate = df[pipeline_module.TARGET].mean()

    assert not df.duplicated().any()
    assert 0.10 < rate < 0.20  # desbalanceamento esperado (~15% de compras)
    assert set(df[pipeline_module.TARGET].unique()) == {0, 1}
