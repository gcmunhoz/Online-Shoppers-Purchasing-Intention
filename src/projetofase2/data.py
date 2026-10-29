"""Carga, limpeza e divisão do dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from projetofase2.config import CATEGORICAL_COLS, FEATURE_COLS, TARGET


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Garante os tipos esperados pelo pipeline, sem alterar o DataFrame original.

    Alvo e ``Weekend`` viram 0/1; categóricas viram texto, para que IDs
    (ex.: ``Region``) não sejam tratados como números ordenados.
    """
    df = df.copy()
    df[TARGET] = df[TARGET].astype(int)
    df["Weekend"] = df["Weekend"].astype(int)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas duplicadas e ajusta os tipos."""
    deduplicated = df.drop_duplicates().reset_index(drop=True)
    return cast_types(deduplicated)


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Lê o CSV bruto e devolve os dados limpos."""
    raw = pd.read_csv(path)
    df = clean_data(raw)
    print(f"Linhas: {len(raw)} -> {len(df)} após remover duplicatas.")
    return df


def load_split(path: str | Path) -> pd.DataFrame:
    """Lê um split já processado (train/test), restaurando os tipos."""
    return cast_types(pd.read_csv(path))


def split_train_test(
    df: pd.DataFrame, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide em treino e teste, estratificando pelo alvo."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state
    )
    return train_df, test_df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa as features (X) do alvo (y)."""
    return df[FEATURE_COLS], df[TARGET]
