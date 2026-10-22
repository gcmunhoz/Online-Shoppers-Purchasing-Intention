"""Testes do pré-processamento e da estrutura do pipeline (sem tuning)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# --------------------------------------------------------------------------- #
# Estrutura
# --------------------------------------------------------------------------- #
def test_build_pipeline_structure(pipeline_module):
    pipe = pipeline_module.build_pipeline()

    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["prep", "rf"]
    assert isinstance(pipe.named_steps["prep"], ColumnTransformer)
    assert isinstance(pipe.named_steps["rf"], RandomForestClassifier)


def test_random_forest_is_configured_for_imbalance_and_reproducibility(pipeline_module):
    rf = pipeline_module.build_pipeline().named_steps["rf"]

    assert rf.class_weight == "balanced_subsample"
    assert rf.random_state == pipeline_module.RANDOM_STATE


def test_preprocessor_covers_every_feature_column(pipeline_module):
    prep = pipeline_module.build_pipeline().named_steps["prep"]
    used = [col for _, _, cols in prep.transformers for col in cols]

    expected = (pipeline_module.NUMERIC_COLS + pipeline_module.CATEGORICAL_COLS
                + pipeline_module.BINARY_COLS)
    assert sorted(used) == sorted(expected)


# --------------------------------------------------------------------------- #
# Pré-processamento
# --------------------------------------------------------------------------- #
def test_preprocessor_output_is_numeric_without_nans(pipeline_module, xy):
    X, _ = xy
    prep = pipeline_module.build_pipeline().named_steps["prep"]
    out = prep.fit_transform(X)

    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(X)
    assert out.isna().sum().sum() == 0
    assert all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes)


def test_preprocessor_expands_categoricals_with_onehot(pipeline_module, xy):
    X, _ = xy
    prep = pipeline_module.build_pipeline().named_steps["prep"]
    out = prep.fit_transform(X)

    # Colunas categóricas originais deixam de existir como tal...
    assert "Month" not in out.columns
    # ...e aparecem como colunas one-hot (0/1)
    month_cols = [c for c in out.columns if c.startswith("Month_")]
    assert len(month_cols) > 1
    assert set(np.unique(out[month_cols].to_numpy())) <= {0.0, 1.0}
    # Numéricas e binária passam direto
    for col in pipeline_module.NUMERIC_COLS + pipeline_module.BINARY_COLS:
        assert col in out.columns


def test_rare_categories_are_grouped(pipeline_module, xy):
    """TrafficType=20 aparece em 1 linha (<1%): deve cair no grupo 'infrequent'."""
    X, _ = xy
    prep = pipeline_module.build_pipeline().named_steps["prep"]
    out = prep.fit_transform(X)

    assert "TrafficType_20" not in out.columns
    assert any("TrafficType" in c and "infrequent" in c for c in out.columns)


def test_numeric_nans_are_imputed_with_training_median(pipeline_module, xy):
    X, _ = xy
    X = X.copy()
    idx = X.index[:10]
    X.loc[idx, "PageValues"] = np.nan

    prep = pipeline_module.build_pipeline().named_steps["prep"].fit(X)
    out = prep.transform(X)

    assert np.allclose(out.loc[idx, "PageValues"], X["PageValues"].median())


def test_categorical_nans_are_imputed(pipeline_module, xy):
    X, _ = xy
    X = X.copy()
    X.loc[X.index[:10], "VisitorType"] = np.nan

    out = pipeline_module.build_pipeline().named_steps["prep"].fit_transform(X)
    assert out.isna().sum().sum() == 0


def test_unseen_category_at_inference_does_not_break(pipeline_module, small_pipeline, xy):
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
    small_pipeline.fit(X, y)
    proba = small_pipeline.predict_proba(X)

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


def test_pipeline_is_reproducible(pipeline_module, xy):
    X, y = xy
    p1 = pipeline_module.build_pipeline().set_params(rf__n_estimators=25).fit(X, y)
    p2 = pipeline_module.build_pipeline().set_params(rf__n_estimators=25).fit(X, y)

    assert np.allclose(p1.predict_proba(X), p2.predict_proba(X))


def test_pipeline_does_not_modify_input(small_pipeline, xy):
    X, y = xy
    X_before = X.copy(deep=True)
    small_pipeline.fit(X, y)
    small_pipeline.predict_proba(X)

    pd.testing.assert_frame_equal(X, X_before)
