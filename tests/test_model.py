"""Testes do pipeline (estrutura, pré-processamento), tuning, limiar e avaliação."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from projetofase2.config import BINARY_COLS, FEATURE_COLS, NUMERIC_COLS
from projetofase2.model import (
    build_deployable_model,
    build_pipeline,
    choose_threshold,
    evaluate,
    tune,
)

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Estrutura
# --------------------------------------------------------------------------- #
def test_build_pipeline_structure():
    pipe = build_pipeline(RANDOM_STATE)

    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["prep", "rf"]
    assert isinstance(pipe.named_steps["prep"], ColumnTransformer)
    assert isinstance(pipe.named_steps["rf"], RandomForestClassifier)


def test_random_forest_handles_imbalance_and_is_reproducible():
    rf = build_pipeline(RANDOM_STATE).named_steps["rf"]

    assert rf.class_weight == "balanced_subsample"
    assert rf.random_state == RANDOM_STATE


def test_preprocessor_covers_every_feature_column():
    prep = build_pipeline(RANDOM_STATE).named_steps["prep"]
    used = [col for _, _, cols in prep.transformers for col in cols]
    assert sorted(used) == sorted(FEATURE_COLS)


# --------------------------------------------------------------------------- #
# Pré-processamento
# --------------------------------------------------------------------------- #
def test_preprocessor_output_is_numeric_without_nans(xy):
    X, _ = xy
    out = build_pipeline(RANDOM_STATE).named_steps["prep"].fit_transform(X)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(X)
    assert out.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes)


def test_preprocessor_expands_categoricals_with_onehot(xy):
    X, _ = xy
    out = build_pipeline(RANDOM_STATE).named_steps["prep"].fit_transform(X)

    assert "Month" not in out.columns
    month_cols = [c for c in out.columns if c.startswith("Month_")]
    assert len(month_cols) > 1
    assert set(np.unique(out[month_cols].to_numpy())) <= {0.0, 1.0}
    for col in NUMERIC_COLS + BINARY_COLS:
        assert col in out.columns


def test_rare_categories_are_grouped(xy):
    """TrafficType=20 aparece em 1 linha (<1%): deve cair no grupo 'infrequent'."""
    X, _ = xy
    out = build_pipeline(RANDOM_STATE).named_steps["prep"].fit_transform(X)

    assert "TrafficType_20" not in out.columns
    assert any("TrafficType" in c and "infrequent" in c for c in out.columns)


def test_numeric_nans_are_imputed_with_training_median(xy):
    X, _ = xy
    X = X.copy()
    idx = X.index[:10]
    X.loc[idx, "PageValues"] = np.nan

    out = build_pipeline(RANDOM_STATE).named_steps["prep"].fit(X).transform(X)
    assert np.allclose(out.loc[idx, "PageValues"], X["PageValues"].median())


def test_categorical_nans_are_imputed(xy):
    X, _ = xy
    X = X.copy()
    X.loc[X.index[:10], "VisitorType"] = np.nan

    out = build_pipeline(RANDOM_STATE).named_steps["prep"].fit_transform(X)
    assert out.isna().sum().sum() == 0


def test_unseen_category_at_inference_does_not_break(small_pipeline, xy):
    X, y = xy
    small_pipeline.fit(X, y)

    X_new = X.head(20).copy()
    X_new["Browser"] = "999"
    X_new["Region"] = "categoria-nova"
    proba = small_pipeline.predict_proba(X_new)

    assert proba.shape == (20, 2)
    assert np.isfinite(proba).all()


# --------------------------------------------------------------------------- #
# Fit / predict
# --------------------------------------------------------------------------- #
def test_fit_and_predict_proba_are_valid(small_pipeline, xy):
    X, y = xy
    proba = small_pipeline.fit(X, y).predict_proba(X)

    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(small_pipeline.classes_) == [0, 1]


def test_prediction_does_not_depend_on_column_order(small_pipeline, xy):
    X, y = xy
    small_pipeline.fit(X, y)

    original = small_pipeline.predict_proba(X)
    shuffled = small_pipeline.predict_proba(X[list(X.columns[::-1])])
    assert np.allclose(original, shuffled)


def test_pipeline_is_reproducible(xy):
    X, y = xy
    p1 = build_pipeline(RANDOM_STATE).set_params(rf__n_estimators=25).fit(X, y)
    p2 = build_pipeline(RANDOM_STATE).set_params(rf__n_estimators=25).fit(X, y)
    assert np.allclose(p1.predict_proba(X), p2.predict_proba(X))


def test_pipeline_does_not_modify_input(small_pipeline, xy):
    X, y = xy
    X_before = X.copy(deep=True)
    small_pipeline.fit(X, y).predict_proba(X)
    pd.testing.assert_frame_equal(X, X_before)


# --------------------------------------------------------------------------- #
# tune
# --------------------------------------------------------------------------- #
def test_tune_returns_fitted_search(params, xy):
    X, y = xy
    search = tune(build_pipeline(RANDOM_STATE), X, y, params.train, RANDOM_STATE)

    assert isinstance(search.best_estimator_, Pipeline)
    assert 0.0 <= search.best_score_ <= 1.0
    assert set(search.best_params_) == set(params.train.param_distributions)
    assert search.scoring == "average_precision"
    assert search.n_iter == params.train.n_iter


# --------------------------------------------------------------------------- #
# choose_threshold
# --------------------------------------------------------------------------- #
def test_choose_threshold_returns_float_in_unit_interval(small_pipeline, xy):
    X, y = xy
    threshold = choose_threshold(small_pipeline, X, y, cv_splits=3, random_state=RANDOM_STATE)

    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0


def test_choose_threshold_is_deterministic(small_pipeline, xy):
    X, y = xy
    t1 = choose_threshold(small_pipeline, X, y, cv_splits=3, random_state=RANDOM_STATE)
    t2 = choose_threshold(small_pipeline, X, y, cv_splits=3, random_state=RANDOM_STATE)
    assert t1 == t2


def test_choose_threshold_does_not_fit_the_given_pipeline(small_pipeline, xy):
    """cross_val_predict deve clonar: o pipeline recebido continua sem treino."""
    X, y = xy
    choose_threshold(small_pipeline, X, y, cv_splits=3, random_state=RANDOM_STATE)
    assert not hasattr(small_pipeline.named_steps["rf"], "estimators_")


# --------------------------------------------------------------------------- #
# evaluate
# --------------------------------------------------------------------------- #
def test_evaluate_prints_and_returns_metrics(small_pipeline, xy, capsys):
    X, y = xy
    metrics = evaluate(small_pipeline.fit(X, y), X, y, threshold=0.5)
    out = capsys.readouterr().out

    assert "ROC-AUC" in out and "Matriz de confusão" in out
    for key in ["test_roc_auc", "test_pr_auc", "test_f1", "test_precision",
                "test_recall", "test_accuracy"]:
        assert 0.0 <= metrics[key] <= 1.0
    confusion = ["test_tn", "test_fp", "test_fn", "test_tp"]
    assert sum(metrics[key] for key in confusion) == len(X)


def test_evaluate_accepts_extreme_thresholds(small_pipeline, xy):
    X, y = xy
    small_pipeline.fit(X, y)

    assert evaluate(small_pipeline, X, y, threshold=0.0)["test_recall"] == 1.0
    assert evaluate(small_pipeline, X, y, threshold=1.01)["test_tp"] == 0


# --------------------------------------------------------------------------- #
# build_deployable_model
# --------------------------------------------------------------------------- #
def test_deployable_model_applies_threshold(small_pipeline, xy):
    X, y = xy
    small_pipeline.fit(X, y)
    model = build_deployable_model(small_pipeline, 0.3, X, y)

    expected = (small_pipeline.predict_proba(X)[:, 1] >= 0.3).astype(int)
    assert (model.predict(X) == expected).all()
