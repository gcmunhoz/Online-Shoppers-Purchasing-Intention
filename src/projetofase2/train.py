"""Estágio ``train``: tuning, avaliação, rastreamento no MLflow e registro do modelo.

Uso:
    python -m projetofase2.train --params configs/params.yaml [--run-name NOME] [--no-register]
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import joblib
import mlflow
from mlflow import MlflowClient
from sklearn.pipeline import Pipeline

from projetofase2.config import (
    DEFAULT_PARAMS_PATH,
    MlflowSettings,
    Params,
    load_params,
    load_settings,
)
from projetofase2.data import load_split, split_features_target
from projetofase2.model import (
    build_deployable_model,
    build_pipeline,
    choose_threshold,
    evaluate,
    tune,
)
from projetofase2.registry import promote_if_best
from projetofase2.tracking import (
    log_data_context,
    log_evaluation,
    log_model,
    log_tuning,
    setup_mlflow,
)


def save_model(pipeline: Pipeline, threshold: float, path: Path) -> None:
    """Salva o pipeline treinado junto com o limiar de decisão."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": threshold}, path)
    print(f"\nModelo salvo em: {path}")


def save_metrics(metrics: dict[str, float], path: Path) -> None:
    """Salva as métricas em JSON (lido por ``dvc metrics show``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_training(params: Params, settings: MlflowSettings, run_name: str | None = None) -> str:
    """Executa o treino completo dentro de uma run do MLflow e devolve o run_id."""
    train_df = load_split(params.data.train_path)
    test_df = load_split(params.data.test_path)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)
    print(f"Treino: {X_train.shape} | Teste: {X_test.shape} | "
          f"% positivos treino: {y_train.mean():.2%}")

    setup_mlflow(settings.tracking_uri, settings.experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        log_data_context(params, train_df, test_df)

        search = tune(build_pipeline(params.random_state), X_train, y_train,
                      params.train, params.random_state)
        best_pipeline = search.best_estimator_
        log_tuning(search)

        threshold = choose_threshold(best_pipeline, X_train, y_train,
                                     params.train.cv_splits, params.random_state)
        metrics = evaluate(best_pipeline, X_test, y_test, threshold)
        log_evaluation(best_pipeline, X_test, y_test, threshold, metrics)

        save_model(best_pipeline, threshold, params.train.model_path)
        save_metrics({**metrics, "decision_threshold": threshold}, params.train.metrics_path)
        mlflow.log_artifact(str(params.train.model_path), artifact_path="joblib")

        deployable = build_deployable_model(best_pipeline, threshold, X_train, y_train)
        registered_name = settings.model_name if settings.register_model else None
        model_info = log_model(deployable, X_train.head(5), registered_name)

    if settings.register_model:
        promote_if_best(
            MlflowClient(tracking_uri=settings.tracking_uri),
            settings.model_name,
            model_info.registered_model_version,
            settings.selection_metric,
            settings.champion_alias,
        )

    print(f"\nRun MLflow: {run.info.run_id}")
    print(f"Para visualizar: mlflow ui --backend-store-uri {settings.tracking_uri}")
    return run.info.run_id


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS_PATH,
                        help="Arquivo YAML de parâmetros")
    parser.add_argument("--run-name", default=None, help="Nome da run no MLflow")
    parser.add_argument("--no-register", action="store_true",
                        help="Não registra o modelo no Model Registry")
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.no_register:
        settings = replace(settings, register_model=False)
    run_training(load_params(args.params), settings, run_name=args.run_name)


if __name__ == "__main__":
    main()
