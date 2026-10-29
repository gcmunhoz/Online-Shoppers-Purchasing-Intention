"""Integração com o MLflow Tracking: experimento, parâmetros, métricas e artefatos."""

import hashlib
import warnings
from pathlib import Path

import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from mlflow.models.model import ModelInfo
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from projetofase2.config import TARGET, Params

# Tipos que o skops (serializador padrão do MLflow para sklearn) exige que sejam
# declarados confiáveis para salvar/carregar árvores. Seguro aqui porque o modelo
# é gerado pelo próprio pipeline, não vem de uma fonte externa.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree", "numpy.dtype"]


def file_md5(path: str | Path) -> str:
    """MD5 do arquivo. Para um arquivo único coincide com o md5 do ``.dvc``,
    o que liga a run ao exato dataset versionado."""
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Aponta o MLflow para o banco e garante que o experimento exista.

    Com SQLite, os artefatos vão para uma pasta ``artifacts`` ao lado do .db,
    independente do diretório de onde o script foi executado.
    """
    mlflow.set_tracking_uri(tracking_uri)

    artifact_location = None
    if tracking_uri.startswith("sqlite:///"):
        db_path = Path(tracking_uri.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_location = (db_path.parent / "artifacts").resolve().as_uri()

    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)


def log_data_context(params: Params, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Registra de onde vieram os dados e como foram divididos."""
    raw_path = params.data.raw_path
    tags = {"estimator": "RandomForestClassifier", "dataset.file": raw_path.name}
    if raw_path.exists():
        tags["dataset.md5"] = file_md5(raw_path)
    mlflow.set_tags(tags)

    mlflow.log_params({
        "random_state": params.random_state,
        "test_size": params.data.test_size,
        "cv_splits": params.train.cv_splits,
        "n_iter": params.train.n_iter,
        "scoring": params.train.scoring,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_features": train_df.shape[1] - 1,
    })
    mlflow.log_metrics({
        "positive_rate_train": float(train_df[TARGET].mean()),
        "positive_rate_test": float(test_df[TARGET].mean()),
    })
    try:  # linhagem do dataset na UI; não deve derrubar o treino se falhar
        dataset = mlflow.data.from_pandas(
            train_df, source=str(params.data.train_path),
            name="online_shoppers_intention_train", targets=TARGET,
        )
        mlflow.log_input(dataset, context="training")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Não foi possível registrar o dataset no MLflow: {exc}", stacklevel=2)


def log_tuning(search: RandomizedSearchCV) -> None:
    """Registra os melhores hiperparâmetros e os resultados completos da busca."""
    mlflow.log_params(search.best_params_)
    cv_results = pd.DataFrame(search.cv_results_)
    mlflow.log_metrics({
        "cv_score_mean": float(search.best_score_),
        "cv_score_std": float(cv_results.loc[search.best_index_, "std_test_score"]),
    })
    mlflow.log_text(cv_results.drop(columns=["params"]).to_csv(index=False),
                    "tuning/cv_results.csv")


def log_evaluation(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    metrics: dict[str, float],
) -> None:
    """Registra métricas de teste, relatório de classificação e importâncias."""
    mlflow.log_metric("decision_threshold", threshold)
    mlflow.log_metrics(metrics)

    pred = (pipeline.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    mlflow.log_text(classification_report(y_test, pred, digits=3, zero_division=0),
                    "reports/classification_report.txt")

    names = pipeline.named_steps["prep"].get_feature_names_out()
    importances = pd.Series(pipeline.named_steps["rf"].feature_importances_, index=names)
    mlflow.log_text(importances.sort_values(ascending=False).rename("importance")
                    .to_csv(index_label="feature"), "reports/feature_importances.csv")


def log_model(
    model: BaseEstimator, input_example: pd.DataFrame, registered_model_name: str | None
) -> ModelInfo:
    """Loga o modelo final na run e, se ``registered_model_name`` for informado,
    cria uma nova versão dele no Model Registry."""
    return mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=input_example,  # o MLflow infere a signature a partir dele
        registered_model_name=registered_model_name,
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )
