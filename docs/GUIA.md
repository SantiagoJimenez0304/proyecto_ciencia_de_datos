# Buenas Prácticas en Ciencia de Datos e IA

## Propósito del taller

Realizar un **análisis exploratorio de datos (EDA)** completo sobre el dataset del Titanic y, a partir de él, **diseñar, evaluar y guardar un modelo de machine learning** que prediga la supervivencia de un pasajero. Todo se construye como un proyecto mantenible: reproducible, verificable, documentado y preparado para crecer. Se utilizan Python, pandas, NumPy, matplotlib, seaborn, SciPy y scikit-learn. PyTorch no se usa: en datos tabulares pequeños como este, los modelos clásicos son suficientes y más fáciles de explicar; las redes neuronales se incorporan solo cuando el problema las requiere.

> **Resultado esperado**: un proyecto estructurado y organizado siguiendo las mejores prácticas de desarrollo para analítica e IA, que se clonar, instalar, ejecutar (`make pipeline`), probar (`make test`) y entender sin depender de rutas locales ni de pasos manuales ocultos, y que termine produciendo un modelo guardado con sus metadatos.

### Alcance

Al terminar serás capaz de:

1. Organizar un proyecto de datos por responsabilidades (ingesta, ingeniería de datos, agregación, modelado) con la arquitectura **medallion** (Bronze, Silver, Gold).
2. Inspeccionar un dataset nuevo y **diagnosticar datos faltantes** con tablas, mapas de calor y un reporte de calidad automático.
3. Construir **variables nuevas** (*feature engineering*) como **funciones puras con pruebas**: `Person`, `Alone`, `TCabin`, `TTitulo`.
4. Elegir y construir el gráfico adecuado con Seaborn y **justificar** estrategias de imputación reconociendo el sesgo que introducen.
5. Distinguir entre **descripción**, **asociación** y **causalidad**, y comunicar con honestidad las limitaciones de tu análisis.
6. Formular un problema de clasificación, definir métricas y **baselines**, y evitar la **fuga de datos** (*data leakage*) con `Pipeline` y `ColumnTransformer`.
7. Comparar modelos con **validación cruzada**, ajustar hiperparámetros, analizar errores e interpretar el modelo.
8. Evaluar **una sola vez** sobre un conjunto de prueba, entrenar el modelo final y **guardarlo** con metadatos que permitan auditarlo.

## 1. Contexto del problema

Imagina que eres analista de datos en una **empresa de seguros** que investiga patrones históricos de supervivencia en desastres marítimos. La gerencia quiere saber, a partir del caso del Titanic (1912), **qué características de una persona aumentaron o disminuyeron su probabilidad de sobrevivir**, para alimentar modelos de riesgo y protocolos de evacuación.

> **Pregunta Central (EDA)**: ¿Qué factores tuvieron más influencia sobre la supervivencia de los pasajeros del Titanic?
> **Pregunta Predictiva (ML)**: Dadas las características de un pasajero conocidas al embarcar, ¿con qué probabilidad habría sobrevivido, y qué tan confiable es esa estimación?

**¿Por qué EDA antes del modelo?** Porque antes de predecir hay que *entender*. El análisis exploratorio revela errores y vacíos en los datos, genera hipótesis y evita construir modelos sobre supuestos falsos. En este dataset **faltan muchos datos** (edad, cabina) y la forma de rellenarlos **cambia las conclusiones**.

### Diccionario de datos

| Variable | Descripción | Tipo esperado |
|---|---|---|
| `PassengerId` | Identificador del pasajero | entero |
| `Survived` | 0 = no sobrevivió, 1 = sobrevivió | entero (categórica binaria) |
| `Pclass` | Clase del tiquete: 1 = alta, 2 = media, 3 = baja | entero (categórica ordinal) |
| `Name` | Nombre completo (incluye el título: Mr, Mrs, Miss…) | texto |
| `Sex` | `male` / `female` | categórica |
| `Age` | Edad en años | decimal (con faltantes) |
| `SibSp` | Nº de hermanos/cónyuge a bordo | entero |
| `Parch` | Nº de padres/hijos a bordo | entero |
| `Ticket` | Número de tiquete | texto |
| `Fare` | Tarifa pagada | decimal |
| `Cabin` | Cabina (la letra indica la cubierta) | texto (muchos faltantes) |
| `Embarked` | Puerto de embarque: C = *Cherbourg*, Q = *Queenstown*, S = *Southampton* | categórica |

> **Nota sobre los datos.** El dataset contiene pasajeros de quienes **sabemos** si sobrevivieron (primeras 891 filas, "etiquetados") y pasajeros de quienes **no** (el resto, "sin etiqueta").

## 2. Principios de diseño

### 2.1 Organizar el código por responsabilidad

Cada módulo tiene una responsabilidad clara. La ingesta obtiene los datos, la ingeniería de datos los valida y transforma, la agregación produce tablas de consumo y el módulo de modelos entrena y evalúa. `job.py` coordina esas piezas, pero no contiene la lógica. Las transformaciones se escriben como **funciones puras** (reciben un `DataFrame`, devuelven uno nuevo, no modifican la entrada).

**Por qué es buena práctica**
- Facilita localizar y corregir errores.
- Permite probar cada transformación sin ejecutar el flujo completo.
- Reduce duplicación y acoplamiento entre módulos.
- Hace posible reutilizar una transformación en más de un flujo.

### 2.2 Separar código, datos, configuración y documentación

El código fuente se versiona; los datos generados y los modelos (archivos pesados o derivados) no. La configuración no sensible (URL, semilla, tamaño del conjunto de prueba) se mantiene fuera de la lógica, en `.env` y `config.py`. Las decisiones y contratos se documentan junto al flujo al que pertenecen.

**Por qué es buena práctica**
- El repositorio no crece con datos derivados ni binarios.
- Varias personas pueden trabajar con el mismo código.
- Los parámetros que afectan la ejecución son explícitos.
- Se puede auditar qué se esperaba recibir y qué produce cada flujo.

### 2.3 Aplicar la arquitectura medallion con criterio

Usamos tres capas locales:

- **Bronze**: copia de los datos tal como llegan (CSV), con modificaciones mínimas y un archivo de metadatos de ingesta (fuente, fecha, huella SHA-256).
- **Silver**: datos validados, tipados, normalizados y enriquecidos (Parquet). Aquí viven las variables derivadas y es la entrada tanto del EDA como del modelado.
- **Gold**: tablas agregadas orientadas a un consumo concreto: el informe de hallazgos.

Las capas describen el estado y el propósito de los datos, no necesariamente tecnologías distintas. En un entorno real, la retención, el formato y los controles de acceso se definen según el proyecto.

### 2.4 Diseñar para reproducibilidad

Una ejecución debe poder repetirse con las mismas entradas y parámetros: fuentes documentadas, dependencias fijadas, semillas aleatorias en la configuración y cero rutas absolutas. Los modelos guardados incluyen versiones de librerías y huella de los datos de entrenamiento.

**Por qué es buena práctica**
- Los resultados se comparan entre ejecuciones y personas.
- Los errores se reproducen.
- Las evaluaciones de modelos son más confiables.

### 2.5 Verificar con pruebas automáticas

Cada módulo trae pruebas en `tests/`. Una prueba describe en código lo que la función **promete** (por ejemplo: "un pasajero de 2 años es `child`"). En este taller las pruebas ya están escritas: tu trabajo es implementar las funciones hasta que pasen (`make test`).

**Por qué es buena práctica**
- Detectas regresiones al cambiar una regla de limpieza.
- Las pruebas documentan decisiones (¿qué pasa con un pasajero sin edad?).
- Permiten refactorizar con confianza.

### 2.6 Evaluar sin fuga de datos

La fuga de datos ocurre cuando información del conjunto de evaluación influye en el entrenamiento (por ejemplo, imputar la edad con el promedio de **todos** los pasajeros antes de dividir). Se evita ajustando todo el preprocesamiento **solo con datos de entrenamiento**, dentro de un `Pipeline`, y abriendo el conjunto de prueba **una única vez**.

**Por qué es buena práctica**
- Las métricas reportadas son estimaciones honestas del desempeño real.
- Evita decisiones basadas en el conjunto de prueba (que dejaría de ser "prueba").


## 3. Estructura del monorepo

Un único repositorio Git reúne los flujos y los componentes compartidos, manteniendo separados los límites de cada flujo.

```text
template_dsai/
├── README.md
├── pyproject.toml
├── .gitignore
├── .env.example
├── Makefile
├── data/                         # Ignorado por Git (se crea al ejecutar)
│   ├── bronze/                   # Copia exacta como llegan los datos + metadatos de ingesta
│   ├── silver/                   # Datos limpios, listos para procesos analíticos y de IA
│   └── gold/                     # Tablas agregadas, listas para informes y tableros
├── models/                       # Modelos entrenados y sus metadatos (ignorado por Git)
├── notebooks/                    # Exploración; no lógica productiva
│   ├── 02_exploracion.ipynb      # Toma datos de Bronze/Silver para hacer el EDA
│   └── 04_modelado.ipynb         # Diseño, entrenamiento y evaluación de diferentes modelos
├── flows/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── paths.py
│   │   ├── logging.py
│   │   └── hashing.py            # Huella SHA-256 de archivos (compartida)
│   ├── job.py                    # Orquesta la ejecución de todo el pipeline
│   ├── 00_ingestion.py           # Desde la fuente a Bronze
│   ├── 01_data_engineering.py    # Prepara los datos de Bronze a Silver
│   ├── 03_aggregation.py         # Realiza agregaciones de Silver a Gold
│   └── 05_ml_models.py           # Diseño, entrenamiento y evaluación del modelo final
├── tests/
│   ├── test_job.py
│   ├── test_00_ingestion.py
│   ├── test_01_data_engineering.py
│   ├── test_03_aggregation.py
│   └── test_05_ml_models.py
└── reports/                      # Conclusiones, ficha del modelo, figuras; normalmente ignoradas
```

**Por qué es buena práctica:** una estructura estable reduce la carga cognitiva, separa la exploración del código reutilizable y ubica cada artefacto en un lugar predecible.

> **Dos detalles técnicos.**
> 1. Los números del nombre indican el orden de las etapas (la numeración salta 02 y 04 porque esas etapas son cuadernos). Un módulo cuyo nombre empieza por un dígito **no se puede importar con `import`**; se carga con `importlib.import_module("flows.01_data_engineering")`. `job.py`, las pruebas y los cuadernos lo hacen así.
> 2. Los nombres de módulos y cuadernos de la plantilla original tenían erratas (`data_engineeging`, `mnodelado`); aquí están corregidos (`data_engineering`, `modelado`).

## Desarrollo

### Ciclo de trabajo en cada paso

1. Lee la teoría y la **lógica** del paso.
2. Implementa la función indicada (si el paso tiene módulo) y ejecuta `pytest tests/test_01_data_engineering.py -q` hasta que pase.
3. Regenera la capa con `make silver` (o `make gold`).
4. En el cuaderno, **reinicia y ejecuta todo** (*Restart & Run All*) y haz el análisis visual leyendo `data/silver/…`.
5. Haz un *commit* pequeño con un mensaje que diga qué decisión tomaste y por qué.

> **Reglas para los cuadernos.** No contienen lógica productiva: si una celda calcula algo que otro módulo necesitará, se mueve a `flows/`. Siempre importan rutas desde `flows.common.paths` (nunca `C:\...` ni `/home/...`) y deben ejecutarse de arriba abajo sin errores. Limpia las salidas antes del *commit* (`nbstripout`).

---

## Parte 0 — Preparación del proyecto y de la capa Bronze (Pasos 0.1–0.6)

> **Teoría — ¿por qué empezar por la estructura?** Un cuaderno que funciona "en mi máquina" es un proyecto frágil. Antes de analizar, creamos el esqueleto que hace al análisis repetible: dependencias declaradas, configuración externa, rutas centralizadas, registro de eventos (*logging*) y una ingesta que deja constancia de lo que recibió.

### Paso 0.1. Crear el repositorio y el esqueleto

```bash
mkdir template_dsai && cd template_dsai
git init
mkdir -p flows/common notebooks tests reports
touch flows/__init__.py flows/common/__init__.py
python -m venv .venv && source .venv/bin/activate    # En Windows: .venv\Scripts\activate
```

### Paso 0.2. `pyproject.toml`: dependencias y configuración de herramientas

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "template-dsai"
version = "0.1.0"
description = "Taller Titanic: EDA y ML con buenas prácticas"
requires-python = ">=3.11"
dependencies = [
  "pandas>=2.2",
  "numpy>=1.26",
  "matplotlib>=3.8",
  "seaborn>=0.13",
  "scipy>=1.12",
  "scikit-learn>=1.4",
  "pyarrow>=15",          # formato Parquet para Silver y Gold
  "joblib>=1.3",          # guardar el modelo
  "python-dotenv>=1.0",   # leer .env
]

[project.optional-dependencies]
dev = ["pytest>=8", "ruff>=0.4", "jupyterlab>=4", "nbstripout>=0.7"]

[tool.setuptools.packages.find]
include = ["flows*"]

[tool.pytest.ini_options]
testpaths = ["tests"]

[tool.ruff]
line-length = 100
```

Instala en modo editable (`pip install -e ".[dev]"`): así `import flows...` funciona desde pruebas y cuadernos sin tocar `sys.path`. Para congelar las versiones exactas con las que obtuviste tus resultados: `pip freeze > requirements.lock.txt` y súbelo al repositorio.

### Paso 0.3. `.gitignore`, `.env.example` y `Makefile`

`.gitignore`
```text
.venv/
__pycache__/
*.egg-info/
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/
.env
data/
models/
reports/
```

`.env.example` (se versiona; cada persona copia a `.env`, que **no** se versiona)
```text
TITANIC_URL=https://paste.c-net.org/MarvelJinxed
TITANIC_LOCAL_PATH=
TITANIC_N_LABELED=891
RANDOM_STATE=42
TEST_SIZE=0.2
CV_FOLDS=5
```

> **Tip — si algo sale mal al cargar.** La URL es un servicio externo y puede caducar. Si la descarga falla, descarga el archivo, guárdalo fuera del repositorio y pon su ruta en `TITANIC_LOCAL_PATH`. Si obtienes **una sola columna** con todo pegado, el separador no es la coma (en el Taller #1 era `|`): ajusta `read_source` con `sep='|'` o `sep=None, engine='python'`.

`Makefile` (las líneas de receta **deben empezar con un tabulador**, no con espacios)
```make
.PHONY: install ingest silver gold train pipeline test lint clean

install:
	pip install -e ".[dev]"

ingest:
	python -m flows.job --stage ingestion

silver:
	python -m flows.job --stage silver

gold:
	python -m flows.job --stage gold

train:
	python -m flows.job --stage train

pipeline:
	python -m flows.job --stage all

test:
	pytest -q

lint:
	ruff check .

clean:
	rm -rf data models .pytest_cache
```

### Paso 0.4. Módulos comunes (ejemplo resuelto)

Estos archivos ya están completos: léelos con atención, porque son el **estilo** que seguirá el resto del proyecto.

`flows/common/paths.py`: **un único lugar** donde se definen las rutas y los nombres de archivo.
```python
"""Rutas y nombres de archivos del proyecto (sin rutas absolutas)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"

BRONZE_LABELED_NAME = "titanic_labeled.csv"
BRONZE_UNLABELED_NAME = "titanic_unlabeled.csv"
BRONZE_METADATA_NAME = "_ingestion_metadata.json"
SILVER_NAME = "titanic_silver.parquet"          # completo y auditable
SILVER_FINAL_NAME = "titanic_final.parquet"     # numérico, sin columnas originales
```

`flows/common/config.py`: parámetros **no sensibles** leídos del entorno.
```python
"""Configuración del proyecto: valores por defecto + .env opcional."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from flows.common.paths import ROOT

load_dotenv(ROOT / ".env")

DEFAULT_URL = "https://paste.c-net.org/MarvelJinxed"


@dataclass(frozen=True)
class Settings:
    data_url: str
    local_path: str
    n_labeled: int
    random_state: int
    test_size: float
    cv_folds: int


def load_settings() -> Settings:
    return Settings(
        data_url=os.getenv("TITANIC_URL", DEFAULT_URL),
        local_path=os.getenv("TITANIC_LOCAL_PATH", ""),
        n_labeled=int(os.getenv("TITANIC_N_LABELED", "891")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        cv_folds=int(os.getenv("CV_FOLDS", "5")),
    )


settings = load_settings()
```

`flows/common/logging.py`
```python
"""Logger con formato uniforme para todas las etapas."""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

`flows/common/hashing.py`
```python
"""Huella SHA-256 de un archivo: sirve para auditar qué datos se usaron."""
import hashlib
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

> **Teoría — `print` vs. `logging`.** `print` desaparece en cuanto el código corre sin consola (un servidor, un horario). `logging` agrega fecha, nivel (INFO/WARNING/ERROR) y nombre del módulo, y puede redirigirse a archivos.

### Paso 0.5. Ingesta a Bronze (ejemplo resuelto)

En el cuaderno original se leía la fuente dos veces (`nrows=891` y `skiprows=892`). Aquí se descarga **una vez** y se divide, y se conserva una copia sin modificar de cada parte.

`flows/00_ingestion.py`
```python
"""Ingesta: fuente remota -> capa Bronze.

Bronze conserva los valores tal como llegan. Los metadatos de ingesta van en
un archivo aparte (_ingestion_metadata.json); nunca se alteran los datos.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from flows.common.config import settings
from flows.common.hashing import sha256_of
from flows.common.logging import get_logger
from flows.common.paths import (
    BRONZE,
    BRONZE_LABELED_NAME,
    BRONZE_METADATA_NAME,
    BRONZE_UNLABELED_NAME,
)

log = get_logger(__name__)


def read_source(source: str) -> pd.DataFrame:
    """Lee la fuente completa (URL o ruta local) sin transformar valores."""
    return pd.read_csv(source)


def split_labeled(df: pd.DataFrame, n_labeled: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa los pasajeros con etiqueta (primeras n filas) de los que no la tienen."""
    labeled = df.iloc[:n_labeled].copy()
    unlabeled = df.iloc[n_labeled:].copy().reset_index(drop=True)
    return labeled, unlabeled


def run(source: str | None = None, n_labeled: int | None = None, out_dir: Path = BRONZE) -> dict:
    """Descarga, divide y guarda en Bronze. Devuelve los metadatos de la ingesta."""
    source = source or settings.local_path or settings.data_url
    n_labeled = n_labeled or settings.n_labeled

    log.info("Leyendo fuente: %s", source)
    df = read_source(source)
    labeled, unlabeled = split_labeled(df, n_labeled)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled_path = out_dir / BRONZE_LABELED_NAME
    unlabeled_path = out_dir / BRONZE_UNLABELED_NAME
    labeled.to_csv(labeled_path, index=False)
    unlabeled.to_csv(unlabeled_path, index=False)

    metadata = {
        "source": source,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "rows_labeled": len(labeled),
        "rows_unlabeled": len(unlabeled),
        "columns": list(df.columns),
        "sha256_labeled": sha256_of(labeled_path),
        "sha256_unlabeled": sha256_of(unlabeled_path),
    }
    (out_dir / BRONZE_METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("Bronze listo: %d etiquetados, %d sin etiqueta", len(labeled), len(unlabeled))
    return metadata
```

`flows/job.py`: el orquestador **solo coordina**.
```python
"""Orquestador: ejecuta las etapas del pipeline en orden."""
import argparse
import importlib

from flows.common.logging import get_logger

log = get_logger(__name__)

# Los módulos con nombre que empieza por dígito no se pueden usar con `import`;
# se cargan con importlib a partir de su ruta de paquete.
STAGES = {
    "ingestion": "flows.00_ingestion",
    "silver": "flows.01_data_engineering",
    "gold": "flows.03_aggregation",
    "train": "flows.05_ml_models",
}


def run_stage(name: str):
    module = importlib.import_module(STAGES[name])
    log.info("Inicio de etapa: %s", name)
    result = module.run()
    log.info("Fin de etapa: %s", name)
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*STAGES, "all"], default="all")
    args = parser.parse_args(argv)
    names = list(STAGES) if args.stage == "all" else [args.stage]
    for name in names:
        run_stage(name)


if __name__ == "__main__":
    main()
```

### Paso 0.6. Pruebas de la ingesta y de `job.py`

`tests/test_00_ingestion.py`
```python
import importlib

import pandas as pd

ing = importlib.import_module("flows.00_ingestion")


def _fake_source(tmp_path):
    df = pd.DataFrame(
        {"PassengerId": range(1, 6), "Survived": [0, 1, 1, None, None], "Sex": list("mffmm")}
    )
    path = tmp_path / "fake.csv"
    df.to_csv(path, index=False)
    return str(path), df


def test_split_labeled():
    df = pd.DataFrame({"Survived": [0, 1, None, None]})
    labeled, unlabeled = ing.split_labeled(df, 2)
    assert len(labeled) == 2 and len(unlabeled) == 2
    assert unlabeled["Survived"].isna().all()


def test_run_writes_bronze_and_metadata(tmp_path):
    source, _ = _fake_source(tmp_path)
    out = tmp_path / "bronze"
    meta = ing.run(source=source, n_labeled=3, out_dir=out)
    assert (out / "titanic_labeled.csv").exists()
    assert (out / "titanic_unlabeled.csv").exists()
    assert (out / "_ingestion_metadata.json").exists()
    assert meta["rows_labeled"] == 3 and meta["rows_unlabeled"] == 2
    assert len(meta["sha256_labeled"]) == 64


def test_bronze_is_an_unmodified_copy(tmp_path):
    source, original = _fake_source(tmp_path)
    out = tmp_path / "bronze"
    ing.run(source=source, n_labeled=3, out_dir=out)
    saved = pd.read_csv(out / "titanic_labeled.csv")
    pd.testing.assert_frame_equal(saved, original.iloc[:3].reset_index(drop=True))
```

`tests/test_job.py`
```python
import importlib

from flows import job


def test_stages_are_ordered_and_importable():
    assert list(job.STAGES) == ["ingestion", "silver", "gold", "train"]
    for module_name in job.STAGES.values():
        assert callable(importlib.import_module(module_name).run)


def test_all_runs_stages_in_order(monkeypatch):
    calls = []
    monkeypatch.setattr(job, "run_stage", lambda name: calls.append(name))
    job.main(["--stage", "all"])
    assert calls == ["ingestion", "silver", "gold", "train"]
```

Ejecuta `make ingest` y `pytest tests/test_00_ingestion.py -q`. (`test_job.py` fallará hasta que existan los cuatro módulos; es lo esperado.) Revisa `data/bronze/_ingestion_metadata.json`: es la "partida de nacimiento" de tus datos.

