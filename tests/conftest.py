"""
Fixtures compartilhadas pelos testes do pipeline Random Forest (Online Shoppers).

Estrutura esperada do projeto:

    <raiz>/
      datasets/  documents/  models/  notebooks/
      src/projetofase2/{__init__.py, model.py}
      tests/     <- este arquivo

- Adiciona <raiz>/src ao sys.path para importar `projetofase2.model`
  (não precisa instalar o pacote nem criar pytest.ini).
- Gera dados sintéticos com o mesmo esquema do dataset real, para que os testes
  rodem rápido e não dependam do CSV de verdade.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
MODULE_NAME = "projetofase2.model"
MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


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


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def pipeline_module():
    return importlib.import_module(MODULE_NAME)


@pytest.fixture
def shoppers_df():
    return make_shoppers_df()


@pytest.fixture
def shoppers_csv(tmp_path, shoppers_df):
    path = tmp_path / "online_shoppers_intention.csv"
    shoppers_df.to_csv(path, index=False)
    return path


@pytest.fixture
def clean_df(pipeline_module, shoppers_csv):
    """DataFrame já passado por `load_data` (sem duplicatas, tipos corretos)."""
    return pipeline_module.load_data(str(shoppers_csv))


@pytest.fixture
def xy(pipeline_module, clean_df):
    cols = (pipeline_module.NUMERIC_COLS + pipeline_module.CATEGORICAL_COLS
            + pipeline_module.BINARY_COLS)
    return clean_df[cols], clean_df[pipeline_module.TARGET]


@pytest.fixture
def small_pipeline(pipeline_module):
    """Pipeline real, mas com poucas árvores para os testes serem rápidos."""
    return pipeline_module.build_pipeline().set_params(rf__n_estimators=25)


@pytest.fixture
def fast_search(monkeypatch, pipeline_module):
    """Troca o RandomizedSearchCV por uma versão minúscula (2 combinações, 3 folds).

    Mantém o fluxo real de `tune`/`main`, mas evita as 150 fits do grid original.
    Também vale para quando o script é executado via runpy (`__main__`).
    """
    import sklearn.model_selection as ms

    original = ms.RandomizedSearchCV

    def _fast(estimator, **kwargs):
        kwargs.update(
            param_distributions={"rf__n_estimators": [10, 20], "rf__max_depth": [3, 5]},
            n_iter=2,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
            n_jobs=1,
            verbose=0,
        )
        return original(estimator, **kwargs)

    monkeypatch.setattr(ms, "RandomizedSearchCV", _fast)
    monkeypatch.setattr(pipeline_module, "RandomizedSearchCV", _fast)
    return _fast


@pytest.fixture(autouse=True)
def isolated_mlflow(monkeypatch, tmp_path):
    """Isola o MLflow em todos os testes: cada teste usa um banco SQLite próprio
    dentro de tmp_path (via env MLFLOW_TRACKING_URI), então nenhuma run de teste
    vai parar na pasta mlflow_data/ do projeto. Vale também para o teste de CLI
    (runpy), que lê a mesma variável de ambiente."""
    uri = f"sqlite:///{(tmp_path / 'mlflow_store' / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri