"""Testes da promoção da melhor versão no Model Registry."""
import mlflow
import mlflow.sklearn
import pytest
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier

from projetofase2.registry import get_version_by_alias, promote_if_best
from projetofase2.tracking import setup_mlflow

MODEL_NAME = "modelo_registry"
METRIC = "test_pr_auc"
ALIAS = "champion"


@pytest.fixture
def client(isolated_mlflow) -> MlflowClient:
    setup_mlflow(isolated_mlflow, "exp-registry")
    return MlflowClient(tracking_uri=isolated_mlflow)


def register_version(score: float) -> str:
    """Cria uma run com a métrica `score` e registra um modelo trivial dela."""
    with mlflow.start_run():
        mlflow.log_metric(METRIC, score)
        info = mlflow.sklearn.log_model(DummyClassifier().fit([[0], [1]], [0, 1]),
                                        name="model", registered_model_name=MODEL_NAME)
    return str(info.registered_model_version)


def champion_version(client: MlflowClient) -> str:
    return str(get_version_by_alias(client, MODEL_NAME, ALIAS).version)


def test_first_version_becomes_champion(client):
    version = register_version(0.5)

    assert promote_if_best(client, MODEL_NAME, version, METRIC, ALIAS)
    assert champion_version(client) == version


def test_better_version_replaces_champion(client):
    v1 = register_version(0.5)
    promote_if_best(client, MODEL_NAME, v1, METRIC, ALIAS)
    v2 = register_version(0.7)

    assert promote_if_best(client, MODEL_NAME, v2, METRIC, ALIAS)
    assert champion_version(client) == v2


def test_worse_or_equal_version_keeps_champion(client):
    v1 = register_version(0.7)
    promote_if_best(client, MODEL_NAME, v1, METRIC, ALIAS)

    for score in [0.6, 0.7]:
        candidate = register_version(score)
        assert not promote_if_best(client, MODEL_NAME, candidate, METRIC, ALIAS)
    assert champion_version(client) == v1


def test_missing_alias_returns_none(client):
    register_version(0.5)
    assert get_version_by_alias(client, MODEL_NAME, "nao-existe") is None
