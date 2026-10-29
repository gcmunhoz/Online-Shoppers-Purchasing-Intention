"""Pipeline de pré-processamento + Random Forest: construção, tuning e avaliação."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    FixedThresholdClassifier,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from projetofase2.config import BINARY_COLS, CATEGORICAL_COLS, NUMERIC_COLS, TrainParams


def build_pipeline(random_state: int) -> Pipeline:
    """Monta o pipeline ``prep`` (imputação + one-hot) -> ``rf`` (Random Forest)."""
    # Random Forest não precisa de escalonamento: só imputação por segurança
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01,
                                 sparse_output=False)),
    ])
    binary_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_COLS),
            ("cat", categorical_pipe, CATEGORICAL_COLS),
            ("bin", binary_pipe, BINARY_COLS),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    classifier = RandomForestClassifier(
        class_weight="balanced_subsample",  # lida com o desbalanceamento
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("prep", preprocessor), ("rf", classifier)])


def tune(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    train_params: TrainParams,
    random_state: int,
) -> RandomizedSearchCV:
    """Busca aleatória de hiperparâmetros com validação cruzada estratificada."""
    cv = StratifiedKFold(n_splits=train_params.cv_splits, shuffle=True,
                         random_state=random_state)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=train_params.param_distributions,
        n_iter=train_params.n_iter,
        scoring=train_params.scoring,
        cv=cv,
        n_jobs=train_params.n_jobs,
        random_state=random_state,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"\nMelhores parâmetros: {search.best_params_}")
    print(f"{train_params.scoring} (CV): {search.best_score_:.4f}")
    return search


def choose_threshold(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_splits: int,
    random_state: int,
) -> float:
    """Escolhe o limiar que maximiza o F1 usando predições out-of-fold no treino."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    oof_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv,
                                  method="predict_proba", n_jobs=-1)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_train, oof_proba)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best = float(thresholds[np.argmax(f1)])
    print(f"Limiar ótimo (F1 OOF): {best:.3f}")
    return best


def evaluate(
    pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float
) -> dict[str, float]:
    """Imprime as métricas no teste e as devolve (chaves prontas para o MLflow)."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "test_pr_auc": float(average_precision_score(y_test, proba)),
        "test_f1": float(f1_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_tn": tn, "test_fp": fp, "test_fn": fn, "test_tp": tp,
    }

    print("\n=== Avaliação no conjunto de teste ===")
    print(f"ROC-AUC: {metrics['test_roc_auc']:.4f}")
    print(f"PR-AUC:  {metrics['test_pr_auc']:.4f}")
    print(f"F1 (limiar {threshold:.3f}): {metrics['test_f1']:.4f}")
    print("\nMatriz de confusão:\n", cm)
    print("\n", classification_report(y_test, pred, digits=3, zero_division=0))
    return metrics


def build_deployable_model(
    pipeline: Pipeline, threshold: float, X_train: pd.DataFrame, y_train: pd.Series
) -> FixedThresholdClassifier:
    """Embute o limiar no modelo, para que ``predict`` use o limiar escolhido
    (e não o corte padrão de 0,5). O pipeline já treinado não é retreinado."""
    model = FixedThresholdClassifier(
        FrozenEstimator(pipeline), threshold=threshold, response_method="predict_proba"
    )
    return model.fit(X_train, y_train)
