"""Testes de tuning, escolha de limiar, avaliação e do fluxo completo (main/CLI)."""
import runpy
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from conftest import make_shoppers_df


# --------------------------------------------------------------------------- #
# tune
# --------------------------------------------------------------------------- #
def test_tune_returns_fitted_search(pipeline_module, fast_search, xy):
    X, y = xy
    search = pipeline_module.tune(pipeline_module.build_pipeline(), X, y)

    assert isinstance(search.best_estimator_, Pipeline)
    assert 0.0 <= search.best_score_ <= 1.0
    assert all(k.startswith("rf__") for k in search.best_params_)
    assert search.best_estimator_.predict_proba(X).shape == (len(X), 2)


def test_tune_uses_average_precision(pipeline_module, fast_search, xy):
    X, y = xy
    search = pipeline_module.tune(pipeline_module.build_pipeline(), X, y)
    assert search.scoring == "average_precision"


# --------------------------------------------------------------------------- #
# choose_threshold
# --------------------------------------------------------------------------- #
def test_choose_threshold_returns_float_in_unit_interval(pipeline_module, small_pipeline, xy):
    X, y = xy
    thr = pipeline_module.choose_threshold(small_pipeline, X, y)

    assert isinstance(thr, float)
    assert 0.0 <= thr <= 1.0


def test_choose_threshold_is_deterministic(pipeline_module, small_pipeline, xy):
    X, y = xy
    t1 = pipeline_module.choose_threshold(small_pipeline, X, y)
    t2 = pipeline_module.choose_threshold(small_pipeline, X, y)

    assert t1 == t2


def test_choose_threshold_does_not_fit_the_given_pipeline(pipeline_module, small_pipeline, xy):
    """cross_val_predict deve clonar: o pipeline recebido continua sem treino."""
    X, y = xy
    pipeline_module.choose_threshold(small_pipeline, X, y)

    assert not hasattr(small_pipeline.named_steps["rf"], "estimators_")


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def test_evaluate_prints_expected_metrics(pipeline_module, small_pipeline, xy, capsys):
    X, y = xy
    small_pipeline.fit(X, y)
    pipeline_module.evaluate(small_pipeline, X, y, threshold=0.5)
    out = capsys.readouterr().out

    assert "ROC-AUC" in out
    assert "F1" in out
    assert "Matriz de confusão" in out
    assert "precision" in out and "recall" in out  # classification_report


def test_evaluate_accepts_extreme_thresholds(pipeline_module, small_pipeline, xy, capsys):
    X, y = xy
    small_pipeline.fit(X, y)

    pipeline_module.evaluate(small_pipeline, X, y, threshold=0.0)  # tudo positivo
    pipeline_module.evaluate(small_pipeline, X, y, threshold=1.0)  # quase tudo negativo
    assert "ROC-AUC" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# main (fluxo completo)
# --------------------------------------------------------------------------- #
def test_main_creates_model_file_and_missing_folders(pipeline_module, fast_search,
                                                     shoppers_csv, tmp_path, capsys):
    out_path = tmp_path / "pasta" / "inexistente" / "modelo.joblib"
    pipeline_module.main(str(shoppers_csv), str(out_path))

    assert out_path.exists()
    assert "Modelo salvo em" in capsys.readouterr().out


def test_main_saved_artifact_is_usable(pipeline_module, fast_search, shoppers_csv, tmp_path):
    out_path = tmp_path / "modelo.joblib"
    pipeline_module.main(str(shoppers_csv), str(out_path))

    artifact = joblib.load(out_path)
    assert set(artifact) == {"pipeline", "threshold"}
    assert isinstance(artifact["threshold"], float)
    assert 0.0 <= artifact["threshold"] <= 1.0

    # Inferência em sessões novas, passando pelo mesmo `load_data`
    new_csv = tmp_path / "novas.csv"
    make_shoppers_df(n=200, seed=7, n_duplicates=0).to_csv(new_csv, index=False)
    new_df = pipeline_module.load_data(str(new_csv))
    cols = (pipeline_module.NUMERIC_COLS + pipeline_module.CATEGORICAL_COLS
            + pipeline_module.BINARY_COLS)

    proba = artifact["pipeline"].predict_proba(new_df[cols])[:, 1]
    decisions = (proba >= artifact["threshold"]).astype(int)

    assert len(decisions) == len(new_df)
    assert set(np.unique(decisions)) <= {0, 1}


def test_main_split_is_stratified_and_has_no_leakage(pipeline_module, fast_search,
                                                     shoppers_csv, tmp_path, monkeypatch):
    seen = {}
    original_choose = pipeline_module.choose_threshold
    original_evaluate = pipeline_module.evaluate

    def spy_choose(best_pipe, X_train, y_train):
        seen["train_idx"], seen["y_train"] = set(X_train.index), y_train
        return original_choose(best_pipe, X_train, y_train)

    def spy_evaluate(best_pipe, X_test, y_test, threshold):
        seen["test_idx"], seen["y_test"] = set(X_test.index), y_test
        return original_evaluate(best_pipe, X_test, y_test, threshold)

    monkeypatch.setattr(pipeline_module, "choose_threshold", spy_choose)
    monkeypatch.setattr(pipeline_module, "evaluate", spy_evaluate)

    pipeline_module.main(str(shoppers_csv), str(tmp_path / "m.joblib"))

    n_total = len(pipeline_module.load_data(str(shoppers_csv)))
    train_idx, test_idx = seen["train_idx"], seen["test_idx"]

    assert not (train_idx & test_idx), "treino e teste não podem compartilhar linhas"
    assert len(train_idx) + len(test_idx) == n_total
    assert abs(len(test_idx) / n_total - 0.2) < 0.01
    assert abs(seen["y_train"].mean() - seen["y_test"].mean()) < 0.02  # estratificado


def test_main_is_reproducible(pipeline_module, fast_search, shoppers_csv, tmp_path):
    out1, out2 = tmp_path / "a.joblib", tmp_path / "b.joblib"
    pipeline_module.main(str(shoppers_csv), str(out1))
    pipeline_module.main(str(shoppers_csv), str(out2))

    a, b = joblib.load(out1), joblib.load(out2)
    assert a["threshold"] == pytest.approx(b["threshold"], abs=1e-6)
    assert a["pipeline"].get_params()["rf__n_estimators"] == b["pipeline"].get_params()["rf__n_estimators"]


# --------------------------------------------------------------------------- #
# CLI (executa o script como __main__, com --csv e --out)
# --------------------------------------------------------------------------- #
def test_cli_runs_end_to_end(pipeline_module, fast_search, shoppers_csv, tmp_path, monkeypatch):
    out_path = tmp_path / "cli" / "modelo.joblib"
    monkeypatch.setattr(sys, "argv", ["model.py",
                                      "--csv", str(shoppers_csv),
                                      "--out", str(out_path)])

    runpy.run_path(str(Path(pipeline_module.__file__)), run_name="__main__")

    assert out_path.exists()
    assert set(joblib.load(out_path)) == {"pipeline", "threshold"}
