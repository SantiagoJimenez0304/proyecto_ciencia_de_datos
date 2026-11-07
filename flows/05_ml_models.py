"""Modelado: Silver -> modelo entrenado y guardado."""
from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from flows.common.config import settings
from flows.common.hashing import sha256_of
from flows.common.logging import get_logger
from flows.common.paths import MODELS, SILVER, SILVER_NAME

log = get_logger(__name__)

TARGET = "Survived"
NUM_FEATURES = ["Age", "SibSp", "Parch", "Fare", "Familia"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked", "TTitulo", "Alone"]
CABIN_NUM = ["HasCabin"]            # evaluacion en Paso 61
CABIN_CAT = ["TCabin"]
CANDIDATE_COLUMNS = NUM_FEATURES + CAT_FEATURES + CABIN_NUM + CABIN_CAT
PRIMARY_METRIC = "roc_auc"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Carga Silver usando AgeOriginal (con nulos) para evitar fuga de datos."""
    silver_path = Path(path) if path else SILVER / SILVER_NAME
    df = pd.read_parquet(silver_path)
    X = df[CANDIDATE_COLUMNS].copy()
    if "AgeOriginal" in df.columns:
        X["Age"] = df["AgeOriginal"]
    return X, df[TARGET].astype(int)


def split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa train y test de forma estratificada (Paso 41)."""
    test_size = test_size if test_size is not None else settings.test_size
    random_state = random_state if random_state is not None else settings.random_state
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


class GroupAgeImputer(BaseEstimator, TransformerMixin):
    """Paso 44: Imputa la edad segun el grupo (Sex, Pclass) aprendido solo en train."""

    def __init__(self, group_cols=("Sex",), target="Age"):
        self.group_cols = group_cols
        self.target = target

    def fit(self, X, y=None):
        cols = list(self.group_cols)
        # Calcula el promedio de edad por cada grupo en train
        self.group_means_ = X.groupby(cols)[self.target].mean()
        # Promedio global por si aparece un grupo nuevo en test
        self.global_mean_ = X[self.target].mean()
        return self

    def transform(self, X):
        X = X.copy()
        cols = list(self.group_cols)
        means = self.group_means_.rename("_fill").reset_index()
        # Unimos para rellenar respetando el indice original
        merged = X[cols].reset_index().merge(means, on=cols, how="left").set_index("index")
        merged = merged.reindex(X.index)
        fill = merged["_fill"].to_numpy(dtype=float)
        # Si no habia promedio para ese grupo, usamos la media global
        fill = np.where(np.isnan(fill), self.global_mean_, fill)
        X[self.target] = X[self.target].fillna(pd.Series(fill, index=X.index))
        return X


def build_pipeline(model, num_features=None, cat_features=None) -> Pipeline:
    """Paso 45: Une preprocesamiento y modelo en un unico Pipeline."""
    num_features = NUM_FEATURES if num_features is None else num_features
    cat_features = CAT_FEATURES if cat_features is None else cat_features

    # 1. Pipeline numerico: mediana para nulos y escalado estandar
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 2. Pipeline categorico: moda para nulos y One-Hot Encoding
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)),
    ])

    # 3. ColumnTransformer para procesar cada columna con su pipeline
    prep = ColumnTransformer(
        [
            ("num", num_pipe, num_features),
            ("cat", cat_pipe, cat_features),
        ],
        sparse_threshold=0,  # salida como arreglo numpy denso
    )

    # 4. Pipeline completo: imputa edad por sexo/clase, preprocesa y entrena el modelo
    return Pipeline([
        ("age_imputer", GroupAgeImputer(group_cols=("Sex", "Pclass"))),
        ("prep", prep),
        ("model", model),
    ])


def get_candidates(random_state: int = 42) -> dict:
    """Paso 47: Catalogo de modelos candidatos con semilla fija."""
    return {
        "dummy": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000, random_state=random_state),
        "tree": DecisionTreeClassifier(max_depth=4, random_state=random_state),
        "forest": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=1),
        "hgb": HistGradientBoostingClassifier(random_state=random_state),
        "knn": KNeighborsClassifier(),
    }


def cross_validate_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    num_features: list[str] | None = None,
    cat_features: list[str] | None = None,
    random_state: int | None = None,
    cv_folds: int | None = None,
) -> pd.DataFrame:
    """Paso 48: Evalua todos los modelos candidatos con validacion cruzada estratificada."""
    random_state = random_state if random_state is not None else settings.random_state
    cv_folds = cv_folds if cv_folds is not None else settings.cv_folds

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    candidates = get_candidates(random_state)

    rows = []
    for name, model in candidates.items():
        pipe = build_pipeline(model, num_features=num_features, cat_features=cat_features)
        res = cross_validate(
            pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True
        )
        row = {"model": name}
        for metric in SCORING:
            row[f"test_{metric}_mean"] = float(res[f"test_{metric}"].mean())
            row[f"test_{metric}_std"] = float(res[f"test_{metric}"].std())
            row[f"train_{metric}_mean"] = float(res[f"train_{metric}"].mean())
            row[f"train_{metric}_std"] = float(res[f"train_{metric}"].std())

        # Brecha de sobreajuste (train - val) para ROC-AUC
        row["gap_roc_auc"] = row["train_roc_auc_mean"] - row["test_roc_auc_mean"]
        rows.append(row)

    df_results = pd.DataFrame(rows).sort_values(f"test_{PRIMARY_METRIC}_mean", ascending=False)
    log.info("Validacion cruzada lista para %d modelos candidatos", len(candidates))
    return df_results


def get_param_grids() -> dict[str, dict]:
    """Paso 56: Espacios de hiperparametros para ajustar los modelos finalistas."""
    return {
        "logreg": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
        },
        "forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [4, 6, 8, None],
            "model__min_samples_leaf": [1, 3, 5],
        },
        "hgb": {
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [2, 3, 4],
            "model__max_iter": [100, 200],
            "model__l2_regularization": [0.0, 1.0],
        },
    }


def evaluate_on_test(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.5) -> dict:
    """Paso 62: Evalua en prueba devolviendo metricas serializables a JSON."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_test": len(X_test),
        "threshold": float(threshold),
    }


# --- Decisiones congeladas (resultado del cuaderno 04 y reports/problema_ml.md) ---
MODEL_NAME = "titanic_model_v1"
FINAL_ESTIMATOR = "hgb"
FINAL_PARAMS = {
    "model__learning_rate": 0.1,
    "model__max_depth": 4,
    "model__max_iter": 200,
    "model__l2_regularization": 1.0,
}
USE_CABIN_FEATURES = False
THRESHOLD = 0.50


def train_final(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Paso 65: Entrena el pipeline definitivo con los parametros congelados."""
    num = NUM_FEATURES + (CABIN_NUM if USE_CABIN_FEATURES else [])
    cat = CAT_FEATURES + (CABIN_CAT if USE_CABIN_FEATURES else [])
    base_model = get_candidates(settings.random_state)[FINAL_ESTIMATOR]
    pipe = build_pipeline(base_model, num_features=num, cat_features=cat)
    pipe.set_params(**FINAL_PARAMS)
    return pipe.fit(X, y)


def save_model(
    pipeline: Pipeline, metadata: dict, name: str = MODEL_NAME, models_dir: Path = MODELS
) -> Path:
    """Paso 66: Guarda el pipeline (.joblib) y sus metadatos (.metadata.json)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.joblib"
    joblib.dump(pipeline, model_path)
    meta = {
        **metadata,
        "name": name,
        "saved_at": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "model_sha256": sha256_of(model_path),
    }
    (models_dir / f"{name}.metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    log.info("Modelo guardado en %s", model_path)
    return model_path


def load_model(path: Path) -> tuple[Pipeline, dict]:
    """Carga el pipeline guardado y sus metadatos de auditoria."""
    path = Path(path)
    pipeline = joblib.load(path)
    meta = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if meta.get("sklearn_version") != sklearn.__version__:
        log.warning(
            "Modelo entrenado con scikit-learn %s; versión actual %s",
            meta.get("sklearn_version"), sklearn.__version__
        )
    return pipeline, meta


def predict(pipeline: Pipeline, X: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Paso 67: Genera probabilidades y predicciones 0/1 para nuevos pasajeros."""
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pd.DataFrame({"proba_survive": proba, "pred": pred}, index=X.index)


def run(in_dir: Path | None = None, models_dir: Path = MODELS) -> Path:
    """Paso 68: Ejecuta Silver -> Evaluacion honesta en test -> Modelo final guardado."""
    silver_path = (Path(in_dir) / SILVER_NAME) if in_dir else (SILVER / SILVER_NAME)
    X, y = load_dataset(silver_path)
    X_tr, X_te, y_tr, y_te = split(X, y)

    # 1. Evaluacion honesta sobre conjunto de prueba
    pipe_eval = train_final(X_tr, y_tr)
    metrics = evaluate_on_test(pipe_eval, X_te, y_te, threshold=THRESHOLD)
    log.info(
        "Evaluación en Test (n=%d): ROC-AUC=%.4f, Exactitud=%.4f, F1=%.4f",
        metrics["n_test"], metrics["roc_auc"], metrics["accuracy"], metrics["f1"]
    )

    # 2. Entrenamiento definitivo con todo el dataset
    final_pipe = train_final(X, y)

    # 3. Metadatos completos
    metadata = {
        "estimator": FINAL_ESTIMATOR,
        "params": FINAL_PARAMS,
        "use_cabin_features": USE_CABIN_FEATURES,
        "features_num": NUM_FEATURES + (CABIN_NUM if USE_CABIN_FEATURES else []),
        "features_cat": CAT_FEATURES + (CABIN_CAT if USE_CABIN_FEATURES else []),
        "target": TARGET,
        "threshold": THRESHOLD,
        "random_state": settings.random_state,
        "n_train": len(X),
        "data_sha256": sha256_of(silver_path),
        "metrics_test": metrics,
    }

    model_path = save_model(final_pipe, metadata, models_dir=models_dir)
    log.info("Entrenamiento final exitoso: modelo guardado en %s", model_path)
    return model_path



