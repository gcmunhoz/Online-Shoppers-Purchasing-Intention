"""Testes da carga, limpeza e divisão dos dados."""
import pandas as pd
import pytest

from projetofase2.config import (
    CATEGORICAL_COLS,
    DEFAULT_PARAMS_PATH,
    FEATURE_COLS,
    NUMERIC_COLS,
    TARGET,
    load_params,
)
from projetofase2.data import (
    cast_types,
    load_raw_data,
    load_split,
    split_features_target,
    split_train_test,
)


# --------------------------------------------------------------------------- #
# load_raw_data
# --------------------------------------------------------------------------- #
def test_load_raw_data_removes_duplicates(shoppers_df, shoppers_csv, capsys):
    df = load_raw_data(shoppers_csv)

    assert len(df) == len(shoppers_df.drop_duplicates())
    assert len(df) < len(shoppers_df)
    assert not df.duplicated().any()
    assert "->" in capsys.readouterr().out  # log de antes -> depois


def test_load_raw_data_resets_index(clean_df):
    assert list(clean_df.index) == list(range(len(clean_df)))


def test_target_and_weekend_are_int(clean_df):
    for col in [TARGET, "Weekend"]:
        assert pd.api.types.is_integer_dtype(clean_df[col])
        assert set(clean_df[col].unique()) <= {0, 1}


def test_categoricals_are_strings(clean_df):
    for col in CATEGORICAL_COLS:
        assert all(isinstance(v, str) for v in clean_df[col]), f"{col} deveria ser str"
    assert "1" in set(clean_df["Region"])  # IDs numéricos viram texto


def test_numeric_values_are_kept(shoppers_df, clean_df):
    expected = shoppers_df.drop_duplicates().reset_index(drop=True)
    for col in NUMERIC_COLS:
        pd.testing.assert_series_equal(clean_df[col], expected[col], check_dtype=False,
                                       check_exact=False, rtol=1e-9)


def test_cast_types_does_not_modify_input(shoppers_df):
    before = shoppers_df.copy(deep=True)
    cast_types(shoppers_df)
    pd.testing.assert_frame_equal(shoppers_df, before)


def test_missing_target_raises(shoppers_df, tmp_path):
    path = tmp_path / "sem_alvo.csv"
    shoppers_df.drop(columns=[TARGET]).to_csv(path, index=False)
    with pytest.raises(KeyError):
        load_raw_data(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "nao_existe.csv")


# --------------------------------------------------------------------------- #
# Divisão treino/teste
# --------------------------------------------------------------------------- #
def test_split_is_stratified_and_has_no_leakage(clean_df):
    train_df, test_df = split_train_test(clean_df, test_size=0.2, random_state=42)

    assert not set(train_df.index) & set(test_df.index)
    assert len(train_df) + len(test_df) == len(clean_df)
    assert abs(len(test_df) / len(clean_df) - 0.2) < 0.01
    assert abs(train_df[TARGET].mean() - test_df[TARGET].mean()) < 0.02


def test_split_is_reproducible(clean_df):
    a, _ = split_train_test(clean_df, test_size=0.2, random_state=42)
    b, _ = split_train_test(clean_df, test_size=0.2, random_state=42)
    assert list(a.index) == list(b.index)


def test_split_features_target(clean_df):
    X, y = split_features_target(clean_df)

    assert list(X.columns) == FEATURE_COLS
    assert y.name == TARGET


def test_load_split_restores_types_after_csv_roundtrip(clean_df, tmp_path):
    path = tmp_path / "split.csv"
    clean_df.to_csv(path, index=False)
    reloaded = load_split(path)

    pd.testing.assert_frame_equal(reloaded, clean_df, check_dtype=False)
    assert all(isinstance(v, str) for v in reloaded["Region"])


# --------------------------------------------------------------------------- #
# Dataset real (pulado automaticamente se o CSV não estiver no caminho)
# --------------------------------------------------------------------------- #
@pytest.fixture
def real_csv():
    path = load_params(DEFAULT_PARAMS_PATH).data.raw_path
    if not path.exists():
        pytest.skip(f"Dataset real não encontrado em {path} (rode `dvc pull`)")
    return path


def test_real_dataset_schema(real_csv):
    raw = pd.read_csv(real_csv)
    expected = FEATURE_COLS + [TARGET]

    assert set(expected) <= set(raw.columns)
    assert len(raw) > 10_000
    assert raw[expected].isna().sum().sum() == 0


def test_real_dataset_after_load(real_csv):
    df = load_raw_data(real_csv)
    rate = df[TARGET].mean()

    assert not df.duplicated().any()
    assert 0.10 < rate < 0.20  # desbalanceamento esperado (~15% de compras)
