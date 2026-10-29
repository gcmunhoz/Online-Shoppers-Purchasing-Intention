"""Testes do estágio train: fluxo completo, artefatos, MLflow Tracking e Registry."""
import json
from dataclasses import replace
from types import SimpleNamespace

import joblib
import mlflow
import numpy as np
import pytest
from conftest import make_shoppers_df
from mlflow import MlflowClient

from projetofase2.config import FEATURE_COLS
from projetofase2.data import load_raw_data
from projetofase2.preprocess import run_preprocess
from projetofase2.registry import get_version_by_alias
from projetofase2.train import main, run_training


@pytest.fixture
def processed(params):
    run_preprocess(params)
    return params


@pytest.fixture
def trained(processed, settings):
    """Roda o treino uma vez e devolve a run já lida do MLflow."""
    run_id = run_training(processed, settings)
    client = MlflowClient(tracking_uri=settings.tracking_uri)
    return SimpleNamespace(run_id=run_id, run=client.get_run(run_id), client=client,
                           params=processed, settings=settings)


def versions(client, settings):
    return client.search_model_versions(f"name='{settings.model_name}'")


# --------------------------------------------------------------------------- #
# Artefatos locais (saídas do estágio no DVC)
# --------------------------------------------------------------------------- #
def test_saves_model_with_threshold(trained):
    artifact = joblib.load(trained.params.train.model_path)

    assert set(artifact) == {"pipeline", "threshold"}
    assert 0.0 <= artifact["threshold"] <= 1.0
    assert trained.run.data.metrics["decision_threshold"] == pytest.approx(artifact["threshold"])


def test_saves_metrics_json(trained):
    metrics = json.loads(trained.params.train.metrics_path.read_text(encoding="utf-8"))

    assert {"test_pr_auc", "test_roc_auc", "test_f1", "decision_threshold"} <= set(metrics)
    assert metrics["test_pr_auc"] == pytest.approx(trained.run.data.metrics["test_pr_auc"])


def test_saved_model_predicts_new_data(trained, tmp_path):
    artifact = joblib.load(trained.params.train.model_path)
    new_csv = tmp_path / "novas.csv"
    make_shoppers_df(n=200, seed=7, n_duplicates=0).to_csv(new_csv, index=False)
    X = load_raw_data(new_csv)[FEATURE_COLS]

    proba = artifact["pipeline"].predict_proba(X)[:, 1]
    assert set(np.unique((proba >= artifact["threshold"]).astype(int))) <= {0, 1}


# --------------------------------------------------------------------------- #
# MLflow Tracking
# --------------------------------------------------------------------------- #
def test_run_finishes_in_configured_experiment(trained):
    assert trained.run.info.status == "FINISHED"
    experiment = trained.client.get_experiment(trained.run.info.experiment_id)
    assert experiment.name == trained.settings.experiment_name


def test_run_logs_expected_params(trained):
    logged = trained.run.data.params

    for key in ["random_state", "test_size", "cv_splits", "n_iter", "scoring",
                "n_train", "n_test", "n_features"]:
        assert key in logged, f"param ausente: {key}"
    assert logged["random_state"] == "42"
    assert logged["n_features"] == str(len(FEATURE_COLS))
    assert any(k.startswith("rf__") for k in logged)  # hiperparâmetros escolhidos


def test_run_logs_expected_metrics(trained):
    metrics = trained.run.data.metrics

    for key in ["test_roc_auc", "test_pr_auc", "test_f1", "test_precision", "test_recall",
                "test_accuracy", "decision_threshold", "cv_score_mean",
                "positive_rate_train", "positive_rate_test"]:
        assert key in metrics, f"métrica ausente: {key}"
        assert 0.0 <= metrics[key] <= 1.0, f"{key} fora de [0, 1]"
    total = metrics["test_tn"] + metrics["test_fp"] + metrics["test_fn"] + metrics["test_tp"]
    assert total == int(trained.run.data.params["n_test"])


def test_run_tags_the_exact_dataset(trained):
    from projetofase2.tracking import file_md5

    tags = trained.run.data.tags
    assert tags["dataset.md5"] == file_md5(trained.params.data.raw_path)


def test_run_logs_artifacts(trained):
    def paths(folder):
        return {f.path for f in trained.client.list_artifacts(trained.run_id, folder)}

    assert "reports/classification_report.txt" in paths("reports")
    assert "reports/feature_importances.csv" in paths("reports")
    assert "tuning/cv_results.csv" in paths("tuning")
    assert f"joblib/{trained.params.train.model_path.name}" in paths("joblib")


def test_failed_training_marks_run_as_failed(processed, settings, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("falha de teste")

    monkeypatch.setattr("projetofase2.train.choose_threshold", boom)
    with pytest.raises(RuntimeError):
        run_training(processed, settings)

    client = MlflowClient(tracking_uri=settings.tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name)
    runs = client.search_runs([experiment.experiment_id])
    assert [r.info.status for r in runs] == ["FAILED"]


# --------------------------------------------------------------------------- #
# Model Registry
# --------------------------------------------------------------------------- #
def test_model_is_registered_and_promoted(trained):
    assert len(versions(trained.client, trained.settings)) == 1
    champion = get_version_by_alias(trained.client, trained.settings.model_name,
                                    trained.settings.champion_alias)
    assert champion is not None and champion.run_id == trained.run_id


def test_champion_model_loads_and_applies_threshold(trained, tmp_path):
    mlflow.set_tracking_uri(trained.settings.tracking_uri)
    uri = f"models:/{trained.settings.model_name}@{trained.settings.champion_alias}"
    model = mlflow.sklearn.load_model(uri)

    new_csv = tmp_path / "novas.csv"
    make_shoppers_df(n=200, seed=7, n_duplicates=0).to_csv(new_csv, index=False)
    X = load_raw_data(new_csv)[FEATURE_COLS]

    threshold = trained.run.data.metrics["decision_threshold"]
    expected = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
    assert (model.predict(X) == expected).all()


def test_no_register_skips_registry(processed, settings):
    run_training(processed, replace(settings, register_model=False))

    client = MlflowClient(tracking_uri=settings.tracking_uri)
    assert len(versions(client, settings)) == 0


def test_each_run_creates_a_new_version(processed, settings):
    run_training(processed, settings)
    run_training(processed, settings)

    client = MlflowClient(tracking_uri=settings.tracking_uri)
    assert len(versions(client, settings)) == 2


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_accepts_run_name_and_no_register(processed, params_file, settings):
    main(["--params", str(params_file), "--run-name", "run-cli", "--no-register"])

    client = MlflowClient(tracking_uri=settings.tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name)
    runs = client.search_runs([experiment.experiment_id])
    assert [r.data.tags["mlflow.runName"] for r in runs] == ["run-cli"]
    assert len(versions(client, settings)) == 0
    assert processed.train.model_path.exists()


def test_no_stray_mlflow_folders_in_cwd(processed, settings, tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)

    run_training(processed, settings)

    assert not (work / "mlruns").exists()
    assert not (work / "mlartifacts").exists()
