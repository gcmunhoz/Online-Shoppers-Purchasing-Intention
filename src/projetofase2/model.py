"""
Pipeline de pré-processamento de dados + Random Forest
Dataset: Online Shoppers Purchasing Intetion

Uso:
    python pipeline_rf_online_shoppers.py --csv online_shoppers_intention.csv"""

import os
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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
CAMINHO_DATASET = str((BASE_DIR / "../../datasets/online_shoppers_intention.csv").resolve())
CAMINHO_MODELO = str((BASE_DIR / "../../models/rf_online_shoppers.joblib").resolve())

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

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
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
        n_iter=30,
        scoring="average_precision",
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
    proba = best_pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)

    print("\n== Avaliação no conjunto de teste ===")
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.4f}")
    print(f"F1 (limiar {threshold:.3f}): {f1_score(y_test, pred):.4f}")
    print("\nMatriz de confusão:\n", confusion_matrix(y_test, pred))
    print("\n", classification_report(y_test, pred, digits=3))

def main(csv_path: str, out_path: str) -> None:
    df = load_data(csv_path)
    X = df[NUMERIC_COLS + CATEGORICAL_COLS + BINARY_COLS]
    y = df[TARGET]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Treino: {X_train.shape} | Teste: {X_test.shape} | "
          f"% positivos treino: {y_train.mean():.2%}")
 
    search = tune(build_pipeline(), X_train, y_train)
    best_pipe = search.best_estimator_
 
    threshold = choose_threshold(best_pipe, X_train, y_train)
    evaluate(best_pipe, X_test, y_test, threshold)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    joblib.dump({"pipeline": best_pipe, "threshold": threshold}, out_path)
    print(f"\nModelo salvo em: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=CAMINHO_DATASET)
    parser.add_argument("--out", default=CAMINHO_MODELO)
    args = parser.parse_args()
    main(args.csv, args.out)