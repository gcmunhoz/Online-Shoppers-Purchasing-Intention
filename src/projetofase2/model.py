"""
Pipeline de pré-processamento de dados + Random Forest
Dataset: Online Shoppers Purchasing Intetion

Uso:
    python pipeline_rf_online_shoppers.py --csv online_shoppers_intention.csv"""

import os
import argparse
import hashlib
import warnings
from typing import Optional
from pathlib import Path
import mlflow.data
import mlflow.sklearn
from mlflow import MlflowClient
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
TARGET = 'Revenue'

# Caminhos relativos à pasta onde este script está (e não ao diretório de execução)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent[1]
CAMINHO_DATASET = str((BASE_DIR / "../../datasets/online_shoppers_intention.csv").resolve())
CAMINHO_MODELO = str((BASE_DIR / "../../models/rf_online_shoppers.joblib").resolve())

# Mlflow: banco SQLite + artefatos em <raiz>/mlflow_data
# SQLite é necessário para usar o Model Registry
MLFLOW_DIR = PROJECT_ROOT / "mlflow_data"
MLFLOW_TRACKING_URI = f"sqlite:///{(MLFLOW_DIR / 'mlflow.db').as_posix()}"
MLFLOW_EXPERIMENT = "online-shoppers-purchase-intention"
MLFLOW_MODEL_NAME = "rf_online_shoppers"

# Colunas numéricas
NUMERIC_COLS = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
]

# Colunas categóricas
CATEGORICAL_COLS = [
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType",
]
 
BINARY_COLS = ["Weekend"]


# =======================================
# Dados e pipeline
# =======================================

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df.attrs["n_rows_raw"] = n_before
    print(f"Linhas: {n_before} -> {len(df)} após remover duplicatas.")

    # Garante tipos corretos
    df[TARGET] = df[TARGET].astype(int)
    df['Weekend'] = df['Weekend'].astype(int)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(str) # evita tratar IDs como números ordenados
    return df

def build_pipeline() -> Pipeline:
    # Random Forest não precisa de escalonamento: só imputação por segurança
    numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median'))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=0.01,
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

    model = RandomForestClassifier(
        class_weight="balanced_subsample", #lida com o desbalanceamento
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("prep", preprocessor), ("rf", model)])


# =====================================
# Treino e avaliação
# =====================================


def tune(pipe: Pipeline, X_train, y_train) -> RandomizedSearchCV:
    param_dist = {
        "rf__n_estimators": [200, 300, 500, 800],
        "rf__max_depth": [None, 8, 12, 16, 24],
        "rf__min_samples_split": [2, 5, 10, 20],
        "rf__min_samples_leaf": [1, 2, 4, 8],
        "rf__max_features": ["sqrt", "log2", 0.3, 0.5],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring=SCORING,
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)
    print(f"\nMelhores parâmetros: {search.best_params_}")
    print(f"PR-AUC (CV): {search.best_score_:.4f}")
    return search

def choose_threshold(best_pipe: Pipeline, X_train, y_train) -> float:
    """
    Escolhe o limiar que maximiza F1 usando predições out-of-fold no treino
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(best_pipe, X_train, y_train, cv=cv,
                            method="predict_proba", n_jobs=-1)[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, oof)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best = thr[np.argmax(f1)]
    print(f"Limiar ótimo (F1 OOF): {best:.3f}")
    return float(best)

def evaluate(best_pipe: Pipeline, X_test, y_test, threshold: float) -> None:
    """
    Imprime as métricas e as devolve em um dict (chaves prontas parao Mlflow).
    """
    proba = best_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    print("\n== Avaliação no conjunto de teste ===")
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
    print(f"F1 (limiar {threshold:.3f}): {f1_score(y_test, pred):.4f}")
    print("\nMatriz de confusão:\n", confusion_matrix(y_test, pred))
    print("\n", classification_report(y_test, pred, digits=3))
    return {
        "test_roc_auc": float(roc_auc),
        "test_pr_auc": float(pr_auc),
        "test_f1": float(f1),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_tn": tn, "test_fp": fp, "test_fn": fn, "test_tp": tp,
    }

# ============================================
# MLflow
# ============================================

def file_md5(path) -> str:
    """md5 do arquivo. Para um arquivo único coincide com o md5 do .dvc,
    o que liga a run ao exato dataset usado."""
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Aponta o MLflow para o banco e garante que o experimento exista.
 
    Com SQLite, os artefatos vão para uma pasta `artifacts` ao lado do .db,
    independente do diretório de onde o script foi executado.
    """
    mlflow.set_tracking_uri(tracking_uri)
 
    artifact_location = None
    if tracking_uri.startswith("sqlite:///"):
        db_path = Path(tracking_uri[len("sqlite:///"):])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_location = (db_path.parent / "artifacts").resolve().as_uri()
 
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)

def log_data_context(df, csv_path, X_train, X_test, y_train, y_test) -> None:
    """Registra de onde vieram os dados e como foram divididos."""
    mlflow.set_tags({
        "dataset.file": Path(csv_path).name,
        "dataset.md5": file_md5(csv_path),
        "estimator": "RandomForestClassifier",
    })
    mlflow.log_params({
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "cv_splits": CV_SPLITS,
        "n_iter": N_ITER,
        "scoring": SCORING,
        "n_rows_raw": df.attrs.get("n_rows_raw", len(df)),
        "n_rows_dedup": len(df),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
    })
    mlflow.log_metrics({
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
    })
    try:  # linhagem do dataset na UI; não deve derrubar o treino se falhar
        dataset = mlflow.data.from_pandas(
            df, source=str(csv_path), name="online_shoppers_intention", targets=TARGET
        )
        mlflow.log_input(dataset, context="training")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"Não foi possível registrar o dataset no MLflow: {exc}")


def log_tuning(search: RandomizedSearchCV) -> None:
    mlflow.log_params(search.best_params_)
    cv_results = pd.DataFrame(search.cv_results_)
    best = search.best_index_
    mlflow.log_metrics({
        "cv_pr_auc_mean": float(search.best_score_),
        "cv_pr_auc_std": float(cv_results.loc[best, "std_test_score"]),
    })
    mlflow.log_text(cv_results.drop(columns=["params"]).to_csv(index=False),
                    "tuning/cv_results.csv")

def log_evaluation(best_pipe, X_test, y_test, threshold: float, metrics: dict) -> None:
    mlflow.log_metric("decision_threshold", threshold)
    mlflow.log_metrics(metrics)
 
    pred = (best_pipe.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    mlflow.log_text(classification_report(y_test, pred, digits=3, zero_division=0),
                    "reports/classification_report.txt")
 
    names = best_pipe.named_steps["prep"].get_feature_names_out()
    importances = pd.Series(best_pipe.named_steps["rf"].feature_importances_, index=names)
    mlflow.log_text(importances.sort_values(ascending=False).rename("importance")
                    .to_csv(index_label="feature"), "reports/feature_importances.csv")


def build_deployable_model(best_pipe: Pipeline, threshold: float, X_train, y_train):
    """Embute o limiar no modelo, para que `predict` use o limiar escolhido
    (e não o corte padrão de 0,5). Exige scikit-learn >= 1.6; sem isso, devolve
    o pipeline puro. Retorna (modelo, aplica_limiar)."""
    try:
        from sklearn.frozen import FrozenEstimator
        from sklearn.model_selection import FixedThresholdClassifier
    except ImportError:
        return best_pipe, False
 
    model = FixedThresholdClassifier(
        FrozenEstimator(best_pipe), threshold=threshold, response_method="predict_proba"
    )
    model.fit(X_train, y_train)  # com FrozenEstimator não retreina o pipeline
    return model, True


def log_model(best_pipe, threshold, X_train, y_train, register_model: bool) -> None:
    model, applies_threshold = build_deployable_model(best_pipe, threshold, X_train, y_train)
    mlflow.set_tag("model.applies_threshold", str(applies_threshold))
    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=X_train.head(5),  # o MLflow infere a signature a partir dele
        registered_model_name=MLFLOW_MODEL_NAME if register_model else None,
    )



# ===========================================
# Orquestração
# ===========================================

def main(csv_path: str, out_path: str, tracking_uri: Optional[str] = None,
         experiment_name: str = MLFLOW_EXPERIMENT, run_name: Optional[str] = None,
         register_model: bool = True) -> str:
    """Executa o pipeline completo e devolve o run_id do MLflow.
 
    Ordem de prioridade do tracking URI: argumento > variável de ambiente
    MLFLOW_TRACKING_URI > MLFLOW_TRACKING_URI deste módulo (SQLite na raiz).
    """
    tracking_uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or MLFLOW_TRACKING_URI
 
    df = load_data(csv_path)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS]
    y = df[TARGET]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Treino: {X_train.shape} | Teste: {X_test.shape} | "
          f"% positivos treino: {y_train.mean():.2%}")
 
    setup_mlflow(tracking_uri, experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        log_data_context(df, csv_path, X_train, X_test, y_train, y_test)
 
        search = tune(build_pipeline(), X_train, y_train)
        best_pipe = search.best_estimator_
        log_tuning(search)
 
        threshold = choose_threshold(best_pipe, X_train, y_train)
        metrics = evaluate(best_pipe, X_test, y_test, threshold)
        log_evaluation(best_pipe, X_test, y_test, threshold, metrics)
 
        # Garante que a pasta de destino do modelo exista antes de salvar
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        joblib.dump({"pipeline": best_pipe, "threshold": threshold}, out_path)
        print(f"\nModelo salvo em: {out_path}")
        mlflow.log_artifact(out_path, artifact_path="joblib")
 
        log_model(best_pipe, threshold, X_train, y_train, register_model)
 
    print(f"\nRun MLflow: {run.info.run_id}")
    print(f"Para visualizar: mlflow ui --backend-store-uri {tracking_uri}")
    return run.info.run_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CAMINHO_DATASET)
    parser.add_argument("--out", default=CAMINHO_MODELO)
    parser.add_argument("--tracking-uri", default=None,
                        help="Padrão: env MLFLOW_TRACKING_URI ou SQLite em <raiz>/mlflow_data")
    parser.add_argument("--experiment", default=MLFLOW_EXPERIMENT)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-register", action="store_true",
                        help="Não registra o modelo no Model Registry")
    args = parser.parse_args()
    main(args.csv, args.out, tracking_uri=args.tracking_uri,
         experiment_name=args.experiment, run_name=args.run_name,
         register_model=not args.no_register)