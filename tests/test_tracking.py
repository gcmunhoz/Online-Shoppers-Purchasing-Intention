"""Testes dos utilitários de integração com o MLflow."""
import hashlib

from mlflow import MlflowClient

from projetofase2.tracking import file_md5, setup_mlflow


def test_file_md5_matches_hashlib(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"conteudo de teste" * 1000)
    assert file_md5(path) == hashlib.md5(path.read_bytes()).hexdigest()


def test_setup_mlflow_creates_experiment_with_artifacts_next_to_db(isolated_mlflow):
    setup_mlflow(isolated_mlflow, "exp-novo")
    experiment = MlflowClient(tracking_uri=isolated_mlflow).get_experiment_by_name("exp-novo")

    assert experiment is not None
    assert "mlflow_store" in experiment.artifact_location
    assert experiment.artifact_location.endswith("artifacts")


def test_setup_mlflow_is_idempotent(isolated_mlflow):
    setup_mlflow(isolated_mlflow, "exp-novo")
    setup_mlflow(isolated_mlflow, "exp-novo")
    experiments = MlflowClient(tracking_uri=isolated_mlflow).search_experiments(
        filter_string="name = 'exp-novo'")
    assert len(experiments) == 1
