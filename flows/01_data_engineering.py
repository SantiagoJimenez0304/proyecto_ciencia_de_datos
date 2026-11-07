"""Ingeniería de datos: Bronze -> Silver.

Cada transformación es una función pura: recibe un DataFrame y devuelve uno
nuevo. El orden importa y está fijado en build_silver().
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE,
    BRONZE_LABELED_NAME,
    SILVER,
    SILVER_FINAL_NAME,
    SILVER_NAME,
)

log = get_logger(__name__)

EXPECTED_COLUMNS = [
    "PassengerId", "Survived", "Pclass", "Name", "Sex", "Age",
    "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked",
]
DECKS = list("ABCDEFGTX")            # orden FIJO: X = desconocida (reproducible)
TITLE_MAP = {                        # consolidación de títulos (Paso 26)
    "Master": "Mr", "the Countess": "Mrs", "Dona": "Mrs", "Mme": "Mrs",
    "Lady": "Miss", "Mlle": "Miss", "Ms": "Miss",
}
MAIN_TITLES = ["Mr", "Mrs", "Miss"]


def validate_bronze(df: pd.DataFrame) -> pd.DataFrame:
    """Valida el contrato de entrada para la capa Bronze."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas requeridas en Bronze: {sorted(missing)}")
    if df["PassengerId"].duplicated().any():
        raise ValueError("Existen PassengerId duplicados en Bronze")
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte los tipos de datos de las columnas numéricas."""
    out = df.copy()
    int_cols = ["Pclass", "SibSp", "Parch"]
    for col in int_cols:
        if col in out.columns:
            out[col] = out[col].astype(int)
    if "Survived" in out.columns and not out["Survived"].isna().all():
        out["Survived"] = out["Survived"].astype(int)
    float_cols = ["Age", "Fare"]
    for col in float_cols:
        if col in out.columns:
            out[col] = out[col].astype(float)
    return out


def add_person(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Person': 'child' si Age < 16; si no, el valor de Sex."""
    out = df.copy()
    out["Person"] = np.where(out["Age"] < 16, "child", out["Sex"])
    return out


def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae TCabin y HasCabin tratando nulos como 'X'."""
    out = df.copy()
    cab = out["Cabin"].fillna("X").astype(str)
    out["TCabin"] = cab.str[0]
    out["HasCabin"] = (out["TCabin"] != "X").astype(int)
    return out


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Familia' (SibSp + Parch) y 'Alone'."""
    out = df.copy()
    out["Familia"] = out["SibSp"] + out["Parch"]
    out["Alone"] = np.where(out["Familia"] > 0, "with Family", "Alone")
    return out


def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame:
    """Mapea Survived a 'Yes'/'No' en la columna 'Survivor'."""
    out = df.copy()
    if "Survived" in out.columns and not out["Survived"].isna().all():
        out["Survivor"] = out["Survived"].map({0: "No", 1: "Yes"})
    else:
        out["Survivor"] = None
    return out


def add_title(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae el título social del nombre."""
    out = df.copy()
    out["TTitulo"] = out["Name"].str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    return out


def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Consolida títulos raros y asigna 'Otro' si no está en MAIN_TITLES."""
    out = df.copy()
    out["TTitulo"] = out["TTitulo"].replace(TITLE_MAP)
    out["TTitulo"] = out["TTitulo"].where(out["TTitulo"].isin(MAIN_TITLES), "Otro")
    return out


def impute_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa los faltantes de Embarked con la moda."""
    out = df.copy()
    moda = out["Embarked"].mode()[0]
    out["Embarked"] = out["Embarked"].fillna(moda)
    return out


def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa Age con el promedio de su género, conservando AgeOriginal."""
    out = df.copy()
    out["AgeOriginal"] = out["Age"].copy()
    out["AgeImputada"] = out["Age"].isna()
    promedios = out.groupby("Sex")["Age"].mean()
    out["Age"] = out["Age"].fillna(out["Sex"].map(promedios))
    return out


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica variables categóricas a representaciones numéricas."""
    out = df.copy()
    out["NSex"] = out["Sex"].map({"female": 0, "male": 1})
    out["NEmbarked"] = out["Embarked"].map({"S": 0, "C": 1, "Q": 2})
    deck_map = {d: i for i, d in enumerate(DECKS)}
    out["NCabin"] = out["TCabin"].map(deck_map)
    title_map = {"Mr": 0, "Mrs": 1, "Miss": 2, "Otro": 3}
    out["NTitulo"] = out["TTitulo"].map(title_map)
    return out


def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia sin Name, Sex, Embarked y Cabin."""
    out = df.copy()
    return out.drop(columns=["Name", "Sex", "Embarked", "Cabin"])


def validate_silver(df: pd.DataFrame) -> pd.DataFrame:
    """Verifica que ninguna de las columnas clave contenga valores nulos."""
    required = [
        "Survived", "Pclass", "Age", "Fare", "Embarked", "Person", "Alone",
        "TCabin", "TTitulo", "NSex", "NEmbarked", "NCabin", "NTitulo",
    ]
    if df[required].isna().any().any():
        null_counts = df[required].isna().sum().to_dict()
        raise ValueError(f"La capa Silver contiene nulos inesperados: {null_counts}")
    return df


def build_silver(bronze: pd.DataFrame) -> pd.DataFrame:
    """Orden deliberado: Person usa la edad ORIGINAL, así que va antes de imputar."""
    return (
        bronze
        .pipe(validate_bronze)
        .pipe(cast_types)
        .pipe(add_person)
        .pipe(add_cabin_features)
        .pipe(add_family_features)
        .pipe(add_survivor_label)
        .pipe(add_title)
        .pipe(consolidate_titles)
        .pipe(impute_embarked)
        .pipe(impute_age)
        .pipe(encode_categoricals)
    )


def run(in_dir: Path = BRONZE, out_dir: Path = SILVER) -> Path:
    bronze = pd.read_csv(Path(in_dir) / BRONZE_LABELED_NAME)
    silver = validate_silver(build_silver(bronze))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out_dir / SILVER_NAME, index=False)
    drop_original_columns(silver).to_parquet(out_dir / SILVER_FINAL_NAME, index=False)
    log.info("Silver listo: %d filas, %d columnas", *silver.shape)
    return out_dir / SILVER_NAME
