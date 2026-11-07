"""Agregaciones: Silver -> Gold. Cada funcion devuelve una tabla lista para un informe."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)


def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Paso 28: Estadisticas descriptivas por grupos clave."""
    age_surv = df.groupby("Survived")["Age"].describe().reset_index()
    age_surv.columns = [str(c) for c in age_surv.columns]

    surv_sex = df.groupby("Sex")["Survived"].agg(["mean", "sum", "count"]).reset_index()
    surv_sex.columns = [str(c) for c in surv_sex.columns]

    surv_pclass = df.groupby("Pclass")["Survived"].agg(["mean", "sum", "count"]).reset_index()
    surv_pclass.columns = [str(c) for c in surv_pclass.columns]

    surv_alone = df.groupby("Alone")["Survived"].agg(["mean", "sum", "count"]).reset_index()
    surv_alone.columns = [str(c) for c in surv_alone.columns]

    return {
        "age_by_survival": age_surv,
        "survival_by_sex": surv_sex,
        "survival_by_pclass": surv_pclass,
        "survival_by_alone": surv_alone,
    }


def chi2_sex_survival(df: pd.DataFrame) -> dict:
    """Paso 29: Prueba chi-cuadrado de independencia entre Sex y Survived."""
    tabla = pd.crosstab(df["Sex"], df["Survived"])
    chi2, p_val, dof, _ = stats.chi2_contingency(tabla)
    return {
        "chi2": float(chi2),
        "p_value": float(p_val),
        "dof": int(dof),
    }


def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 34: Tasa de supervivencia cruzando Pclass y Sex."""
    table = df.pivot_table(values="Survived", index="Pclass", columns="Sex", aggfunc="mean").reset_index()
    table.columns = [str(c) for c in table.columns]
    return table


def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35 (1): Tasa y conteo de supervivencia por puerto de embarque."""
    table = df.groupby("Embarked")["Survived"].agg(["mean", "count"]).reset_index()
    table.columns = [str(c) for c in table.columns]
    return table


def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 35 (2): Tasa de supervivencia por puerto estratificada por clase."""
    table = df.pivot_table(values="Survived", index="Embarked", columns="Pclass", aggfunc="mean").reset_index()
    table.columns = [str(c) for c in table.columns]
    return table


def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36 (1): Asigna cuartiles de tarifa calculados dentro de cada clase (sin mutar entrada)."""
    out = df.copy()
    out["FareQ"] = out.groupby("Pclass")["Fare"].transform(
        lambda s: pd.qcut(s, 4, labels=False, duplicates="drop")
    )
    return out


def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 36 (2): Tasa de supervivencia por cuartil de tarifa y clase."""
    df_fq = add_fare_quartile(df)
    table = df_fq.pivot_table(values="Survived", index="FareQ", columns="Pclass", aggfunc="mean").reset_index()
    table.columns = [str(c) for c in table.columns]
    return table


def survival_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 37: Tasa de supervivencia y conteo por tamano de familia."""
    table = df.groupby("Familia")["Survived"].agg(["mean", "count"]).reset_index()
    table.columns = [str(c) for c in table.columns]
    return table


def outlier_mask_iqr(s: pd.Series) -> pd.Series:
    """Paso 38 (1): Mascara booleana para valores atipicos segun la regla de 1.5 * IQR."""
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (s < lower) | (s > upper)


def outlier_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Paso 38 (2): Conteo de atipicos en Fare, Age y Familia, y comparacion de supervivencia."""
    age_col = "AgeOriginal" if "AgeOriginal" in df.columns else "Age"
    fare_mask = outlier_mask_iqr(df["Fare"])
    age_clean = df[age_col].dropna()
    age_mask_clean = outlier_mask_iqr(age_clean)
    age_mask = df.index.isin(age_clean.index[age_mask_clean])
    fam_mask = outlier_mask_iqr(df["Familia"])

    rows = []
    for var, mask in [("Fare", fare_mask), (age_col, pd.Series(age_mask, index=df.index)), ("Familia", fam_mask)]:
        n_outliers = int(mask.sum())
        out_surv = float(df.loc[mask, "Survived"].mean()) if n_outliers > 0 else 0.0
        norm_surv = float(df.loc[~mask, "Survived"].mean())
        rows.append({
            "variable": var,
            "n_outliers": n_outliers,
            "survival_rate_outliers": out_surv,
            "survival_rate_normal": norm_surv,
        })
    res = pd.DataFrame(rows)
    res.columns = [str(c) for c in res.columns]
    return res


def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    """Genera todas las tablas agregadas de la capa Gold a partir de Silver."""
    silver_file = (Path(in_dir) / SILVER_NAME) if in_dir else (SILVER / SILVER_NAME)
    df = pd.read_parquet(silver_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        **group_stats(df),
        "survival_by_class_sex": survival_by_class_sex(df),
        "survival_by_embarked": survival_by_embarked(df),
        "survival_by_embarked_class": survival_by_embarked_class(df),
        "survival_by_fare_quartile_class": survival_by_fare_quartile_class(df),
        "survival_by_family": survival_by_family(df),
        "outlier_summary": outlier_summary(df),
    }

    for name, table in tables.items():
        table_str = table.copy()
        table_str.columns = [str(c) for c in table_str.columns]
        table_str.to_parquet(out_dir / f"{name}.parquet", index=False)

    chi2_results = chi2_sex_survival(df)
    (out_dir / "chi2_sex_survival.json").write_text(
        json.dumps(chi2_results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    return out_dir
