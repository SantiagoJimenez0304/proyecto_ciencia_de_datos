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

def get_logger(name: str, log_file: str = "job.log") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Formateador compartido para consola y archivo
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        
        # 1. Handler para la consola (stdout)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # 2. Handler para el archivo de logs
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
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

---

## Parte 1 — Primer contacto con los datos (Pasos 1–5)

> **Teoría.** Antes de graficar nada, un analista responde cuatro preguntas: *¿cuánto hay?* (dimensiones), *¿de qué tipo es cada columna?* (tipos), *¿cómo se ve?* (muestra) y *¿qué falta?* (nulos). Los datos faltantes no son un detalle: **pueden sesgar todas tus conclusiones**. Esta primera mirada se hace sobre **Bronze**, porque ahí está la verdad sin retocar.

**Dónde:** `notebooks/02_exploracion.ipynb`, sección "1. Diagnóstico sobre Bronze". Celda de preparación (ejemplo resuelto):

```python
# 1) Importamos las librerías con sus alias convencionales
import pandas as pd                 # tablas (DataFrame)
import numpy as np                  # cálculo numérico y condiciones vectorizadas
import matplotlib.pyplot as plt     # gráficos base (Seaborn se apoya en él)
import seaborn as sns               # gráficos estadísticos de alto nivel
from scipy import stats             # pruebas estadísticas (Parte 6)

from flows.common.paths import BRONZE, SILVER, GOLD, BRONZE_LABELED_NAME, SILVER_NAME

# 2) Estilo global: todos los gráficos heredan esta configuración
sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.figsize'] = (8, 5)   # tamaño por defecto (ancho, alto) en pulgadas
plt.rcParams['axes.titlesize'] = 13

# 3) Cargamos Bronze (nunca la URL: la ingesta ya la guardó)
df = pd.read_csv(BRONZE / BRONZE_LABELED_NAME)
```

> **Teoría — ¿por qué varias librerías?** Pandas organiza y transforma tablas; NumPy ofrece operaciones numéricas rápidas; Matplotlib dibuja; Seaborn dibuja gráficos *estadísticos* con pocas líneas (agrupa, calcula intervalos de confianza, aplica temas); SciPy contiene pruebas estadísticas. Se combinan porque son compatibles entre sí.

> **Tip — copia de seguridad.** Como el cuaderno solo *lee* de disco, ya no hay riesgo de "dañar" `df`: si te equivocas, vuelve a ejecutar la celda de carga.

### Paso 1. ¿Cuántos pasajeros (filas) y variables (columnas) hay?

**Lógica:** `shape` devuelve una tupla `(filas, columnas)`.

```python
# Obtén las dimensiones del DataFrame.
# Pista: shape es un ATRIBUTO (no lleva paréntesis).
# Verificación: deberías obtener 891 filas y 12 columnas.

```

### Paso 2. ¿Cuáles son las variables, sus nombres y tipos?

**Lógica:** `info()` muestra nombre, **cantidad de valores no nulos** y tipo (`dtype`) de cada columna. Compara el número de no nulos con el total de filas: ahí se descubren los datos faltantes.

```python
# Muestra el resumen estructural del DataFrame.
# Pista: info() es un MÉTODO (lleva paréntesis) e imprime directamente.

```

> En Pandas 3.x las columnas de texto se muestran como `str`; en versiones anteriores, como `object`. Son equivalentes para este taller.

### Paso 3. Muestra las primeras 10 filas

```python
# Muestra las 10 primeras filas.
# Pista: head(n) recibe el número de filas.

```

### Paso 4. Estadísticas descriptivas

**Lógica:** `describe()` resume las columnas **numéricas** (conteo, media, desviación, mínimo, cuartiles, máximo). Para las de texto hay que pedirlo aparte.

```python
# 1. Estadísticas descriptivas de las columnas numéricas.
# 2. Estadísticas de las columnas NO numéricas.
#    Pista: df.select_dtypes(exclude='number').describe()
#    (mira 'unique' y 'top': ¿cuántos nombres, tiquetes y cabinas distintos hay?)

```

### Paso 5. Identificar datos faltantes (tabla, mapa de calor y reporte automático)

**Lógica:** `df.isnull()` devuelve una tabla de `True/False` (True = falta el dato). Sumarla da el conteo por columna. Un **heatmap** muestra *dónde* faltan y si hay patrones (filas a las que les faltan varias cosas a la vez).

**5a. En el cuaderno (exploración):**
```python
# 1. Cuenta los nulos por columna con isnull().sum().
# 2. Dibuja el mapa de calor de df.isnull():
#    sns.heatmap(..., cbar=False, yticklabels=False, cmap='viridis')
# 3. Añade un título con plt.title(...) y cierra con plt.show().
# Verificación: Age, Cabin y Embarked son las únicas columnas con faltantes.
```

**5b. En el módulo (automatización).** Lo que un analista mira a mano, un pipeline debe registrarlo solo. Añade a `flows/00_ingestion.py` la función `quality_report(df)` y haz que `run()` guarde su resultado en `data/bronze/_quality_report.json`:

```python
def quality_report(df: pd.DataFrame) -> dict:
    """Resumen de calidad: dimensiones, tipos y faltantes por columna.

    Debe devolver un dict con las claves:
      n_rows, n_cols, dtypes ({col: str(dtype)}), nulls ({col: int}), null_pct ({col: float 0-100}).
    """
    # TODO: usar df.shape, df.dtypes, df.isnull().sum()
    raise NotImplementedError
```

Añade a `tests/test_00_ingestion.py`:
```python
def test_quality_report_counts_nulls():
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", None]})
    rep = ing.quality_report(df)
    assert rep["n_rows"] == 3 and rep["n_cols"] == 2
    assert rep["nulls"] == {"a": 1, "b": 1}
    assert round(rep["null_pct"]["a"], 1) == 33.3
```

---

## Parte 2 — Conociendo a los pasajeros (Pasos 6–10)

> **Teoría — dos herramientas nuevas.**
> - `sns.countplot(data=df, x='col', hue='otra')` cuenta filas por categoría; `hue` divide cada barra por una segunda variable.
> - `pd.crosstab(a, b)` produce la **tabla cruzada** (conteos de cada combinación). Con `normalize='columns'` (o `'index'`) los conteos pasan a **proporciones** que suman 1 por columna (o por fila).
>
> Para **comparar grupos de distinto tamaño** usa proporciones, no conteos absolutos.

> **A partir de aquí el cuaderno lee Silver.** Reemplaza la carga por `df = pd.read_parquet(SILVER / SILVER_NAME)`. Silver se va construyendo paso a paso: cada vez que implementes una función en `01_data_engineering.py`, ejecuta `make silver` y recarga el cuaderno.

### Preparación: el esqueleto de `flows/01_data_engineering.py`

Crea el archivo con este esqueleto. Las funciones son **puras** (copian el `DataFrame` de entrada) y `build_silver` las encadena con `.pipe`. Las irás implementando a lo largo de las Partes 2 a 8; `tests/test_01_data_engineering.py` (al final de la Parte 8) dice qué promete cada una.

```python
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
    BRONZE, BRONZE_LABELED_NAME, SILVER, SILVER_FINAL_NAME, SILVER_NAME,
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


def validate_bronze(df: pd.DataFrame) -> pd.DataFrame: ...      # contrato de entrada
def cast_types(df: pd.DataFrame) -> pd.DataFrame: ...           # Survived -> int, etc.
def add_person(df: pd.DataFrame) -> pd.DataFrame: ...           # Paso 8
def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame: ...   # Pasos 11 y 24 (TCabin, HasCabin)
def add_family_features(df: pd.DataFrame) -> pd.DataFrame: ...  # Paso 14 (Familia, Alone)
def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame: ...   # Paso 16 (Survivor)
def add_title(df: pd.DataFrame) -> pd.DataFrame: ...            # Paso 25 (TTitulo)
def consolidate_titles(df: pd.DataFrame) -> pd.DataFrame: ...   # Paso 26
def impute_embarked(df: pd.DataFrame) -> pd.DataFrame: ...      # Paso 21
def impute_age(df: pd.DataFrame) -> pd.DataFrame: ...           # Paso 22
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame: ...  # Pasos 21, 23, 24, 26
def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame: ...  # Paso 32
def validate_silver(df: pd.DataFrame) -> pd.DataFrame: ...      # Paso 33


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
```

> Un cuerpo de función que solo contiene `...` no hace nada: al ejecutar `make silver` no obtendrás columnas nuevas hasta que lo implementes. Empieza por `validate_bronze` y `cast_types` (ver Parte 8 para el contrato) y deja que `build_silver` devuelva el `DataFrame` tal cual mientras avanzas.

### Paso 6. Distribución por género y su cruce con la clase

**Dónde:** cuaderno.
```python
# 1. Gráfico de conteo de 'Sex'.
# 2. Gráfico de conteo de 'Sex' con hue='Pclass'.
# 3. Tabla cruzada Sex x Pclass NORMALIZADA POR COLUMNAS:
#    ¿qué proporción de cada clase son mujeres?

```

### Paso 7. Distribución de edades

**Lógica:** un **histograma** divide la variable numérica en intervalos (*bins*) y cuenta cuántos casos caen en cada uno. `Age` tiene nulos: Seaborn los ignora al graficar, pero es bueno hacerlo explícito con `dropna()`.

**Dónde:** cuaderno.
```python
# 1. Histograma de la edad: sns.histplot(df['Age'].dropna(), bins=30)
#    Añade título y etiquetas de los ejes.
# 2. Calcula la edad promedio (Pandas ignora los NaN automáticamente).
# 3. Explora: mediana, mínimo y máximo de la edad.
```

> En este punto de la ruta `Age` todavía **no está imputada** en Silver (la imputación llega en el Paso 22), así que estás viendo solo a los pasajeros con edad conocida.

### Paso 8. Crear la variable `Person` (hombre, mujer o niño)

**Lógica:** `np.where(condición, valor_si_True, valor_si_False)` crea una columna de forma **vectorizada** (sin `for`). Si `Age < 16` → `'child'`; si no, se conserva el valor de `Sex`.

> **Advertencia — los NaN y las comparaciones.** En Pandas/NumPy, `NaN < 16` da `False`. Por lo tanto, **un pasajero sin edad nunca será `child`**: se queda con su sexo. En el cuaderno original esto ocurría en silencio. Ahora la regla es **explícita, documentada y probada**.

**Dónde:** `flows/01_data_engineering.py` → `add_person(df)`.
```python
def add_person(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Person': 'child' si Age < 16; si no, el valor de Sex.

    Regla para edad desconocida (NaN): se usa Sex. Esto subestima a los niños
    sin edad registrada; el sesgo se cuantifica en el Reto 8.1.
    """
    # TODO: copia df; np.where(df['Age'] < 16, 'child', df['Sex'])
    raise NotImplementedError
```

**Cuaderno:** cuenta cuántos hay de cada tipo con `value_counts()`.

> **Reto 8.1.** ¿Cuántos pasajeros tienen `Age` nula? Si todos ellos fueran niños con la misma proporción que los que sí tienen edad, ¿cuántos `child` "faltarían"? Escribe tu estimación en `reports/conclusiones_eda.md`.

### Paso 9. ¿Cuántos niños viajaban en cada clase?

**Dónde:** cuaderno.
```python
# Tabla cruzada de 'Pclass' (filas) contra 'Person' (columnas).
```

### Paso 10. Gráficos de densidad (KDE) de la edad

> **Teoría — KDE.** La *estimación de densidad por kernel* es un "histograma suavizado": la curva muestra dónde se concentran los valores. Con `hue` se superponen varias curvas. Usa `common_norm=False` para que **cada grupo se normalice por separado** (si no, los grupos grandes "aplastan" a los pequeños).

**Dónde:** cuaderno.
```python
# Crea una figura de 1 fila x 3 columnas: fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
# Dibuja en cada eje (parámetro ax=axes[i]) un sns.kdeplot(data=df, x='Age', hue=..., fill=True, common_norm=False):
#   axes[0] -> hue='Sex'
#   axes[1] -> hue='Person'
#   axes[2] -> hue='Pclass'   (añade palette='viridis')
# Ponle un título a cada eje con axes[i].set_title(...) y cierra con plt.tight_layout(); plt.show()
```

---

## Parte 3 — Ubicación, embarque y composición familiar (Pasos 11–15)

> **Teoría — extraer información de texto.** Con una Serie de texto, el accesor `.str` aplica operaciones a cada elemento a la vez: `serie.str[0]` toma el primer carácter. En `Cabin` (por ejemplo `'C85'`) la primera letra indica la **cubierta**.

### Paso 11. Extraer la cubierta de la cabina

**Lógica:** a diferencia del cuaderno original (que solo exploraba la cubierta), el módulo crea **dos** columnas: `TCabin` (letra de la cubierta; `'X'` si la cabina es desconocida) y `HasCabin` (1 si hay cabina registrada, 0 si no).

**Dónde:** `flows/01_data_engineering.py` → `add_cabin_features(df)`.
```python
def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'TCabin' (primera letra de Cabin, 'X' si es nula) y 'HasCabin' (0/1)."""
    # TODO: df['Cabin'].str[0].fillna('X'); df['Cabin'].notna().astype(int)
    raise NotImplementedError
```

**Cuaderno:** muestra los conteos por cubierta con `value_counts()` de `TCabin`.

> **Decisión de diseño.** Rellenar con `'X'` convierte "no sé" en una categoría explícita (opción *c* de las tres decisiones de limpieza de la Parte 5). `Cabin` original se conserva en Silver: **nunca se descarta información en silencio**.

### Paso 12. Gráfico de conteo por cubierta (con depuración de anómalos)

**Dónde:** cuaderno.
```python
# 1. Dibuja sns.countplot(x=cubierta, order=sorted(cubierta.unique())) con título y etiquetas,
#    usando solo las cabinas conocidas: cubierta = df.loc[df['TCabin'] != 'X', 'TCabin'].
# 2. Si encuentras una cubierta con un solo registro (anómala; en el Titanic suele ser la 'T'),
#    exclúyela SOLO DEL GRÁFICO (cubierta = cubierta[cubierta != 'T']) y vuelve a graficar.
#    No elimines la fila de df: un grupo diminuto es un problema para estadística, no un dato erróneo.
```

### Paso 13. Puerto de embarque y clase

**Dónde:** cuaderno.
```python
# 1. Gráfico de conteo de 'Embarked' con hue='Pclass'.
# 2. Tabla cruzada Embarked x Pclass normalizada por FILAS (normalize='index'):
#    ¿qué composición de clases tiene cada puerto?
```

### Paso 14. Crear la variable `Alone` (solo vs. con familia)

**Lógica:** `SibSp + Parch` es el número de familiares a bordo. Si es 0, viaja solo.

**Dónde:** `flows/01_data_engineering.py` → `add_family_features(df)`.
```python
def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Familia' = SibSp + Parch y 'Alone' ('with Family' si Familia > 0, 'Alone' si no)."""
    # TODO
    raise NotImplementedError
```

**Cuaderno:** calcula la proporción de cada categoría con `value_counts(normalize=True)`.

### Paso 15. Visualizar la distribución de `Alone`

**Dónde:** cuaderno.
```python
# Gráfico de conteo de 'Alone' (con título y etiquetas).
```

---

## Parte 4 — Análisis de supervivencia: el núcleo del EDA (Pasos 16–20)

> **Teoría — la media de una variable 0/1 es una proporción.** Si `Survived` vale 1 para quien sobrevivió y 0 para quien no, entonces `df['Survived'].mean()` es la **tasa de supervivencia**. Por eso los gráficos de Seaborn que promedian (`catplot` tipo punto, `lmplot`) usan la columna **numérica** `Survived` en el eje y; la columna de texto `Survivor` sirve para etiquetar gráficos de conteo.
>
> **Las barritas verticales** de los gráficos de puntos son **intervalos de confianza al 95 %**: indican la *incertidumbre*. Barras largas = pocos datos o mucha variabilidad.

### Paso 16. Panorama general

**Dónde:** `add_survivor_label(df)` en el módulo y el resto en el cuaderno.
```python
def add_survivor_label(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega 'Survivor' mapeando 0 -> 'No' y 1 -> 'Yes' (solo para etiquetar gráficos)."""
    # TODO: .map({0: 'No', 1: 'Yes'})
    raise NotImplementedError
```
```python
# Cuaderno:
# 2. Gráfico de conteo de 'Survivor'.
# 3. Calcula la tasa general de supervivencia (media de 'Survived').
# Verificación: ronda el 38 %.
```

### Paso 17. Supervivencia por clase

**Dónde:** cuaderno.
```python
# Gráfico de puntos: sns.catplot(data=df, x='Pclass', y='Survived', kind='point', height=4, aspect=1.4)
# Añade título con plt.title(...) y etiqueta del eje y: "Tasa de supervivencia".
```

### Paso 18. Supervivencia por clase y tipo de persona

**Dónde:** cuaderno.
```python
# Repite el gráfico anterior añadiendo hue='Person' y hue_order=['child', 'female', 'male'].
```

### Paso 19. Edad y supervivencia

> **Teoría — `lmplot`.** Dibuja una **recta de regresión** (con banda de confianza) de y contra x. Con `y` binaria (0/1) la recta es solo una *aproximación visual* de la tendencia. El parámetro `x_bins` agrupa la edad en intervalos y dibuja la **tasa media por intervalo**, lo cual es mucho más legible.

**Dónde:** cuaderno.
```python
# 1. sns.lmplot(data=df, x='Age', y='Survived')
# 2. Añade hue='Pclass'.
# 3. Define generaciones = [10, 20, 40, 60, 80] y usa x_bins=generaciones con hue='Pclass'.
# 4. Repite el punto 3 con hue='Sex'.
```

### Paso 20. Relación tarifa-edad

**Dónde:** cuaderno.
```python
# 1. Dispersión simple: plt.scatter(df['Age'], df['Fare'], alpha=0.5)
# 2. Desafío: sns.scatterplot(data=df, x='Age', y='Fare', hue='Pclass', alpha=0.5, palette='viridis')
# Añade título y etiquetas.
```

---

## Parte 5 — Limpieza y preparación de datos (Pasos 21–26)

> **Teoría — tres decisiones de limpieza.** Un dato faltante puede (a) eliminarse, (b) rellenarse con un valor "razonable" (*imputación*) o (c) conservarse como una categoría propia ("desconocido"). **Ninguna opción es neutral:** cada una cambia lo que los datos dicen. Documenta siempre qué hiciste y por qué.
>
> **Teoría — codificación.** Los modelos y las correlaciones necesitan números. Convertir categorías a números con `.map({...})` crea columnas nuevas (con prefijo `N`). Cuidado: **un número impone un orden** que quizá no existe (¿es "Q = 2" mayor que "S = 0"?). En la Parte 11 verás una alternativa mejor para modelar (*one-hot*).
>
> **Teoría — imputar en Silver vs. imputar dentro del modelo.** Rellenar la edad con el promedio por género es razonable para **explorar**. Pero calcular ese promedio con *todos* los pasajeros y luego usarlo para entrenar y evaluar un modelo filtra información del conjunto de prueba hacia el entrenamiento (fuga de datos, Principio 2.6). Por eso Silver **conserva la edad original** (`AgeOriginal`) junto con la imputada, y en la Parte 11 el modelo imputará por su cuenta, solo con datos de entrenamiento.

### Paso 21. Datos faltantes en `Embarked`

**Dónde:** `impute_embarked(df)` y, para el punto 4, `encode_categoricals(df)`.
```python
# 1. ¿Cuántos valores faltan en 'Embarked'?
# 2. Calcula la categoría más frecuente: df['Embarked'].mode()[0]
# 3. Rellena los faltantes con esa moda usando fillna (asigna el resultado a la columna;
#    evita inplace=True).
# 4. Crea df['NEmbarked'] mapeando S -> 0, C -> 1, Q -> 2.   (va en encode_categoricals)
```

### Paso 22. Datos faltantes en `Age`

**Lógica:** en lugar de rellenar con el promedio global, rellenamos con el **promedio de su grupo** (género). Se calcula una tabla `promedio por género` y se "traduce" cada pasajero a su promedio con `.map`.

**Dónde:** `flows/01_data_engineering.py` → `impute_age(df)`.
```python
def impute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Imputa 'Age' con el promedio por 'Sex'.

    Conserva la edad original en 'AgeOriginal' (con NaN) y marca las filas
    imputadas en 'AgeImputada' (bool). Usada para EXPLORAR; el modelo (Parte 11)
    vuelve a imputar solo con datos de entrenamiento.
    """
    # 1. df['AgeOriginal'] = df['Age']; df['AgeImputada'] = df['Age'].isna()
    # 2. promedios = df.groupby('Sex')['Age'].mean()
    # 3. df['Age'] = df['Age'].fillna(df['Sex'].map(promedios))
    raise NotImplementedError
```

> **Reto 22.1.** Registra la desviación estándar de la edad **antes y después** de imputar (usa `log.info`). ¿Por qué baja? ¿Qué le pasa a la forma del histograma del Paso 7 si lo repites con `Age` imputada? Anota la conclusión en `reports/conclusiones_eda.md`.

### Paso 23. Convertir `Sex` a número

**Dónde:** `encode_categoricals(df)`: crea `NSex` mapeando `female -> 0` y `male -> 1`.

### Paso 24. Tratamiento de `Cabin`

**Lógica:** la parte de relleno con `'X'` y `TCabin` ya la hizo `add_cabin_features` (Paso 11). Falta la codificación.

**Dónde:** `encode_categoricals(df)`.
```python
# Crea df['NCabin'] con un mapa FIJO sobre la constante DECKS:
#    mapa = {letra: i for i, letra in enumerate(DECKS)}
# (el cuaderno original usaba sorted(df['TCabin'].unique()): si una cubierta desaparecía de
#  los datos, los números cambiaban y los resultados dejaban de ser comparables.)
```

> **Reflexión.** `NCabin` convierte cubiertas en números ordenados (A=0 … X=8). ¿Tiene sentido que "X, desconocida" sea mayor que "G"? Para correlaciones es una simplificación; para modelar usaremos *one-hot*.

### Paso 25. Extracción de títulos del nombre

> **Teoría — expresiones regulares (regex).** Los nombres tienen la forma `"Braund, Mr. Owen Harris"`. El título está entre la coma y el punto. El patrón `r',\s*([^.]+)\.'` significa: una coma, espacios opcionales (`\s*`), y captura (`( )`) todo lo que no sea un punto (`[^.]+`), hasta el punto. `str.extract(patron, expand=False)` devuelve la parte capturada.

**Dónde:** `add_title(df)`.
```python
# 1. Crea df['TTitulo'] = df['Name'].str.extract(r',\s*([^.]+)\.', expand=False).str.strip()
```
**Cuaderno:**
```python
# 2. Analiza la frecuencia de cada título (value_counts).
# 3. Tabla cruzada de título vs. género (pd.crosstab(df['TTitulo'], df['Sex'])).
```

### Paso 26. Consolidar títulos

**Lógica:** los títulos raros dejan grupos de 1 o 2 personas: inútiles para estadística. Se agrupan en categorías más generales con `replace` (un diccionario) y `where/isin` (todo lo que no esté en la lista principal pasa a `'Otro'`).

**Dónde:** `consolidate_titles(df)` (usa las constantes `TITLE_MAP` y `MAIN_TITLES`).
```python
# 1. Reemplaza con el diccionario TITLE_MAP:
#      Master -> Mr ;  the Countess, Dona, Mme -> Mrs ;  Lady, Mlle, Ms -> Miss
# 2. Todo título que NO esté en MAIN_TITLES debe pasar a 'Otro'.
#    Pista: df['TTitulo'].where(df['TTitulo'].isin(MAIN_TITLES), 'Otro')
# 3. En 'encode_categoricals', crea df['NTitulo'] mapeando Mr -> 0, Mrs -> 1, Miss -> 2, Otro -> 3.
```
**Cuaderno:** revisa los conteos finales.

> **Cuestionamiento.** `Master` (niño varón) se fusiona con `Mr`. Esa decisión mezcla niños con adultos: anota en `reports/conclusiones_eda.md` si te parece defendible y qué columna (`Person`) la compensa.

---

## Parte 6 — Correlaciones y estadísticas por grupos (Pasos 27–29)

> **Teoría — correlación de Pearson (repaso del Taller #1).** `df[cols].corr()` produce una **matriz** con la correlación lineal de cada par de variables (−1 a +1). Un **heatmap** con `annot=True` la hace legible. Recuerda: mide relación **lineal**, no implica causalidad, y es poco informativa para variables categóricas codificadas arbitrariamente.

### Paso 27. Matriz de correlación

**Dónde:** cuaderno.
```python
# 1. Elige las variables numéricas relevantes:
#    cols = ['Survived', 'Pclass', 'NSex', 'Age', 'SibSp', 'Parch', 'Fare', 'NEmbarked', 'NCabin', 'NTitulo']
# 2. Calcula corr = df[cols].corr()
# 3. Dibuja sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, vmin=-1, vmax=1)
#    con plt.figure(figsize=(9, 7)).
# 4. Imprime solo la columna de 'Survived' ordenada: corr['Survived'].sort_values()
```

### Paso 28. Estadísticas descriptivas por grupos

> De aquí en adelante empieza la **capa Gold**. Crea `flows/03_aggregation.py` con el esqueleto de la Parte 9 e implementa primero `group_stats`.

**Dónde:** `flows/03_aggregation.py` → `group_stats(df)`, que devuelve un diccionario de tablas:
```python
# Genera cuatro tablas con groupby y devuélvelas en un dict:
# 'age_by_survival':      df.groupby('Survived')['Age'].describe()
# 'survival_by_sex':      df.groupby('Sex')['Survived'].agg(['mean', 'sum', 'count'])
# 'survival_by_pclass':   igual que la anterior con 'Pclass'
# 'survival_by_alone':    igual con 'Alone'
# Pista: en cada tabla usa .reset_index() para que el índice sea una columna normal.
```

### Paso 29. ¿Es la diferencia por sexo estadísticamente significativa? (Chi-cuadrado)

> **Teoría — prueba de independencia chi-cuadrado.** Compara los conteos observados en una tabla cruzada con los que se esperarían **si las dos variables fueran independientes**.
> - **H₀:** `Sex` y `Survived` son independientes (el sexo no se asocia con sobrevivir).
> - **H₁:** sí hay asociación.
>
> Si `p < 0.05` se rechaza H₀. **Un p-valor pequeño no dice que la asociación sea grande ni causal**, solo que es poco probable que sea azar.

**Dónde:** `chi2_sex_survival(df)` en `03_aggregation.py`.
```python
def chi2_sex_survival(df: pd.DataFrame) -> dict:
    """Devuelve {'chi2': float, 'p_value': float, 'dof': int}."""
    # 1. tabla = pd.crosstab(df['Sex'], df['Survived'])
    # 2. chi2, p, gl, esperados = stats.chi2_contingency(tabla)
    raise NotImplementedError
```
En el cuaderno (o en `reports/conclusiones_eda.md`), escribe tu conclusión **con sus límites**: ¿qué dice y qué no dice este p-valor?

---

## Parte 7 — Visualizaciones comparativas avanzadas (Pasos 30–31)

### Paso 30. Boxplots múltiples (cuadrícula 2×2)

> **Teoría — boxplot.** La caja va del cuartil 1 al 3 (el 50 % central); la línea interior es la mediana; los "bigotes" llegan hasta 1.5·IQR; los puntos fuera son **candidatos a atípicos**.

**Dónde:** cuaderno.
```python
# 1. fig, axes = plt.subplots(2, 2, figsize=(12, 9))
# 2. Dibuja sns.boxplot(data=df, x=..., y='Age', ax=axes[f, c]) para:
#      axes[0,0] -> x='Sex'     axes[0,1] -> x='Pclass'
#      axes[1,0] -> x='Embarked' axes[1,1] -> x='Survived'
# 3. Cierra con plt.tight_layout(); plt.show()
# Atención: ¿usas 'Age' original o imputada? Con 'AgeOriginal' ves solo edades reales.
```

### Paso 31. Distribución de la tarifa

**Dónde:** cuaderno.
```python
# 1. Histograma con KDE: sns.histplot(df['Fare'], kde=True, bins=40)
# 2. Calcula media, mediana y asimetría (skew) de 'Fare'.
#    ¿Qué implica una media muy superior a la mediana para un modelo lineal?
```

---

## Parte 8 — Dataset final y verificación (Pasos 32–33)

### Paso 32. Limpieza final

**Lógica:** ya creamos versiones numéricas o derivadas de `Name`, `Sex`, `Embarked` y `Cabin`. Las originales se eliminan **en una copia**, de modo que Silver completo (`titanic_silver.parquet`) conserve todo para el análisis crítico de la Parte 9 y la vista final (`titanic_final.parquet`) quede solo numérica.

**Dónde:** `drop_original_columns(df)`.
```python
def drop_original_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una COPIA sin Name, Sex, Embarked y Cabin."""
    # TODO: df.drop(columns=[...])
    raise NotImplementedError
```

### Paso 33. Verificación (contratos de entrada y de salida)

**Lógica:** una capa es confiable solo si **falla ruidosamente** cuando algo no cumple su contrato.

**Dónde:** `validate_bronze(df)` y `validate_silver(df)`.
```python
# validate_bronze: lanza ValueError si faltan columnas de EXPECTED_COLUMNS
#                  o si hay PassengerId duplicados.
# validate_silver: lanza ValueError si hay nulos en
#    ['Survived','Pclass','Age','Fare','Embarked','Person','Alone','TCabin','TTitulo',
#     'NSex','NEmbarked','NCabin','NTitulo'] (AgeOriginal sí puede tener nulos).
# Devuelven el mismo df cuando todo está bien, para poder usarlas en .pipe().
# cast_types: Survived, Pclass, SibSp, Parch -> int; Age y Fare -> float.
```

**Pruebas** (`tests/test_01_data_engineering.py`). Con estas pruebas verdes, las Partes 2–8 del módulo están terminadas:

```python
import importlib

import numpy as np
import pandas as pd
import pytest

de = importlib.import_module("flows.01_data_engineering")


@pytest.fixture
def sample():
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5, 6],
        "Survived": [0, 1, 1, 0, 1, 0],
        "Pclass": [3, 1, 3, 1, 2, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
            "Heikkinen, Miss. Laina",
            "Palsson, Master. Gosta Leonard",
            "Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)",
            "Moran, Dr. James",
        ],
        "Sex": ["male", "female", "female", "male", "female", "male"],
        "Age": [22.0, 38.0, np.nan, 2.0, 33.0, np.nan],
        "SibSp": [1, 1, 0, 3, 0, 0],
        "Parch": [0, 0, 0, 1, 0, 0],
        "Ticket": list("ABCDEF"),
        "Fare": [7.25, 71.28, 7.92, 21.07, 86.5, 8.46],
        "Cabin": [None, "C85", None, None, "B77", None],
        "Embarked": ["S", "C", "S", None, "S", "Q"],
    })


def test_validate_bronze_rejects_missing_columns(sample):
    with pytest.raises(ValueError):
        de.validate_bronze(sample.drop(columns=["Fare"]))


def test_add_person_is_explicit_about_unknown_age(sample):
    out = de.add_person(sample)
    assert out["Person"].tolist() == ["male", "female", "female", "child", "female", "male"]


def test_family_features(sample):
    out = de.add_family_features(sample)
    assert out["Familia"].tolist() == [1, 1, 0, 4, 0, 0]
    assert out["Alone"].tolist() == ["with Family", "with Family", "Alone",
                                     "with Family", "Alone", "Alone"]


def test_cabin_features(sample):
    out = de.add_cabin_features(sample)
    assert out["TCabin"].tolist() == ["X", "C", "X", "X", "B", "X"]
    assert out["HasCabin"].tolist() == [0, 1, 0, 0, 1, 0]


def test_titles_are_extracted_and_consolidated(sample):
    out = de.consolidate_titles(de.add_title(sample))
    assert out["TTitulo"].tolist() == ["Mr", "Mrs", "Miss", "Mr", "Mrs", "Otro"]


def test_impute_embarked_uses_mode(sample):
    out = de.impute_embarked(sample)
    assert out["Embarked"].isna().sum() == 0
    assert out.loc[3, "Embarked"] == "S"


def test_impute_age_by_sex_keeps_original(sample):
    out = de.impute_age(sample)
    assert out["Age"].isna().sum() == 0
    assert out.loc[2, "Age"] == pytest.approx(35.5)     # promedio de mujeres: (38 + 33) / 2
    assert out.loc[5, "Age"] == pytest.approx(12.0)     # promedio de hombres: (22 + 2) / 2
    assert out["AgeOriginal"].isna().sum() == 2
    assert out["AgeImputada"].tolist() == [False, False, True, False, False, True]


def test_functions_do_not_mutate_their_input(sample):
    before = sample.copy(deep=True)
    for fn in (de.add_person, de.add_family_features, de.add_cabin_features,
               de.impute_embarked, de.impute_age):
        fn(sample)
    pd.testing.assert_frame_equal(sample, before)


def test_build_silver_end_to_end(sample):
    out = de.validate_silver(de.build_silver(sample))
    assert len(out) == len(sample)
    for col in ["Person", "Alone", "TCabin", "TTitulo", "NSex", "NEmbarked", "NCabin",
                "NTitulo", "AgeOriginal", "Survivor"]:
        assert col in out.columns
    assert out.loc[1, "NCabin"] == de.DECKS.index("C")   # el mapa de cubiertas es fijo


def test_validate_silver_fails_on_nulls(sample):
    out = de.build_silver(sample)
    out.loc[0, "Fare"] = np.nan
    with pytest.raises(ValueError):
        de.validate_silver(out)


def test_drop_original_columns(sample):
    out = de.drop_original_columns(sample)
    assert not {"Name", "Sex", "Embarked", "Cabin"} & set(out.columns)
    assert "Name" in sample.columns       # la entrada no se modifica
```

Ejecuta `pytest tests/test_01_data_engineering.py -q`, luego `make silver`, y confirma que `data/silver/` contiene los dos archivos Parquet.

---

## Parte 9 — Análisis crítico: interacciones, asociaciones y atípicos (Pasos 34–38)

Esta es la parte donde pasas de **describir** a **argumentar**. Aquí un número solo no basta: hay que preguntarse si es *grande*, si es *confiable* (¿cuántos pasajeros hay detrás?) y si es *justo* compararlo (¿hay una tercera variable que confunde?).

> **Teoría — variable de confusión.** Si el puerto parece influir en la supervivencia, quizá lo que influye es la **clase** (cada puerto trae pasajeros de clases distintas). Para separar efectos se **estratifica**: se compara dentro de cada nivel de la tercera variable.

**Dónde:** cada paso es una función en `flows/03_aggregation.py` que produce una tabla **Gold**. La interpretación se escribe en el cuaderno y en `reports/conclusiones_eda.md`.

Esqueleto del módulo (completa `group_stats` y `chi2_sex_survival` de la Parte 6 y las funciones de esta parte):

```python
"""Agregaciones: Silver -> Gold. Cada función devuelve una tabla lista para un informe."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from scipy import stats

from flows.common.logging import get_logger
from flows.common.paths import GOLD, SILVER, SILVER_NAME

log = get_logger(__name__)


def group_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]: ...             # Paso 28
def chi2_sex_survival(df: pd.DataFrame) -> dict: ...                          # Paso 29
def survival_by_class_sex(df: pd.DataFrame) -> pd.DataFrame: ...              # Paso 34
def survival_by_embarked(df: pd.DataFrame) -> pd.DataFrame: ...               # Paso 35 (1)
def survival_by_embarked_class(df: pd.DataFrame) -> pd.DataFrame: ...         # Paso 35 (2)
def add_fare_quartile(df: pd.DataFrame) -> pd.DataFrame: ...                  # Paso 36 (1)
def survival_by_fare_quartile_class(df: pd.DataFrame) -> pd.DataFrame: ...    # Paso 36 (2)
def survival_by_family(df: pd.DataFrame) -> pd.DataFrame: ...                 # Paso 37
def outlier_mask_iqr(s: pd.Series) -> pd.Series: ...                          # Paso 38 (1)
def outlier_summary(df: pd.DataFrame) -> pd.DataFrame: ...                    # Paso 38 (2-3)


def run(in_dir: Path | None = None, out_dir: Path = GOLD) -> Path:
    df = pd.read_parquet((in_dir or SILVER) / SILVER_NAME)
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
        table.to_parquet(out_dir / f"{name}.parquet", index=False)
    (out_dir / "chi2_sex_survival.json").write_text(json.dumps(chi2_sex_survival(df), indent=2))
    log.info("Gold listo: %d tablas en %s", len(tables) + 1, out_dir)
    return out_dir
```

> **Gotcha de Parquet.** Parquet exige que **los nombres de columna sean texto**. Un `pivot_table` con `columns='Pclass'` produce columnas `1, 2, 3` (enteros): conviértelas con `.rename(columns=str)` antes de guardar. Todas las tablas deben salir con `.reset_index()` (índice como columna normal).

### Paso 34. Interacciones: clase × género

```python
# survival_by_class_sex: tasa de supervivencia con Pclass en filas y Sex en columnas:
# df.pivot_table(values='Survived', index='Pclass', columns='Sex', aggfunc='mean')
```

### Paso 35. Puerto de embarque y supervivencia (¿con confusión?)

```python
# 1. survival_by_embarked: tasa y número de pasajeros por 'Embarked' (groupby + agg ['mean', 'count']).
# 2. survival_by_embarked_class: repite estratificando por clase:
#    pivot_table con index='Embarked', columns='Pclass'.
```

### Paso 36. Tarifa y supervivencia dentro de cada clase

**Lógica:** para ver si pagar más ayudaba **dentro de una misma clase**, dividimos la tarifa en cuartiles *por clase* con `pd.qcut` y comparamos tasas.

```python
# 1. add_fare_quartile: crea df['FareQ'] con cuartiles de Fare calculados dentro de cada Pclass:
#    df['FareQ'] = df.groupby('Pclass')['Fare'].transform(
#        lambda s: pd.qcut(s, 4, labels=False, duplicates='drop'))
# 2. survival_by_fare_quartile_class: pivot_table de 'Survived' con index='FareQ' y columns='Pclass'.
```

### Paso 37. Tamaño de la familia

```python
# 1. survival_by_family: tasa de supervivencia y cantidad de pasajeros por 'Familia' (groupby + agg).
# 2. Cuaderno: sns.pointplot(data=df, x='Familia', y='Survived').
#    ¿Qué pasa con las barras de error en familias de 6 o más? ¿Qué te dice eso?
```

### Paso 38. Pasajeros atípicos

> **Teoría — regla IQR.** `IQR = Q3 − Q1`. Es atípico lo que queda fuera de `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`. *Atípico no significa erróneo:* antes de eliminar, pregunta si es un error de captura o un caso real.

```python
# 1. outlier_mask_iqr(s): devuelve una máscara booleana (usa s.quantile([0.25, 0.75])).
# 2. outlier_summary: aplícala a 'Fare', 'Age' (usa AgeOriginal) y 'Familia' y cuenta cuántos
#    atípicos hay en cada una; añade la tasa de supervivencia de los atípicos de 'Fare'
#    frente al resto.
# 3. Cuaderno: muestra los pasajeros con la tarifa más alta (nlargest).
```

**Pruebas** (`tests/test_03_aggregation.py`):

```python
import importlib

import pandas as pd
import pytest

agg = importlib.import_module("flows.03_aggregation")


@pytest.fixture
def silver_like():
    return pd.DataFrame({
        "Survived": [1, 1, 0, 0, 1, 0, 0, 1],
        "Pclass": [1, 1, 1, 3, 3, 3, 3, 3],
        "Sex": ["female", "male", "male", "female", "female", "male", "male", "male"],
        "Embarked": ["S", "C", "S", "Q", "S", "S", "C", "Q"],
        "Fare": [80.0, 90.0, 70.0, 7.0, 8.0, 7.5, 9.0, 300.0],
        "Age": [30.0, 40.0, 50.0, 20.0, 25.0, 30.0, 35.0, 28.0],
        "Familia": [0, 1, 0, 2, 0, 0, 1, 0],
        "Alone": ["Alone", "with Family", "Alone", "with Family", "Alone", "Alone",
                  "with Family", "Alone"],
    })


def test_outlier_mask_iqr_flags_only_extreme_values():
    mask = agg.outlier_mask_iqr(pd.Series([1, 2, 3, 4, 5, 100]))
    assert mask.tolist() == [False, False, False, False, False, True]


def test_survival_by_class_sex_is_a_valid_rate_table(silver_like):
    out = agg.survival_by_class_sex(silver_like)
    assert "Pclass" in out.columns
    rates = out.drop(columns="Pclass")
    assert ((rates.dropna() >= 0) & (rates.dropna() <= 1)).all().all()
    assert all(isinstance(c, str) for c in out.columns)      # requisito de Parquet


def test_fare_quartiles_are_computed_within_class(silver_like):
    out = agg.add_fare_quartile(silver_like)
    for _, group in out.groupby("Pclass"):
        assert group["FareQ"].between(0, 3).all()
    assert "FareQ" not in silver_like.columns                 # no muta la entrada


def test_chi2_returns_valid_p_value(silver_like):
    res = agg.chi2_sex_survival(silver_like)
    assert set(res) == {"chi2", "p_value", "dof"}
    assert 0 <= res["p_value"] <= 1


def test_run_writes_gold_tables(tmp_path, silver_like):
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    silver_like.to_parquet(silver_dir / "titanic_silver.parquet", index=False)
    out = agg.run(in_dir=silver_dir, out_dir=tmp_path / "gold")
    assert (out / "survival_by_class_sex.parquet").exists()
    assert (out / "chi2_sex_survival.json").exists()
```

Ejecuta `pytest tests/test_03_aggregation.py -q` y `make gold`.

### Entregable de la Parte 9 — `reports/conclusiones_eda.md`

Escribe un informe corto para la gerencia de la aseguradora con esta estructura:

1. **Hallazgos** (3 a 5), cada uno con su cifra **y su n** (cuántos pasajeros hay detrás).
2. **Decisiones de limpieza y su sesgo** (edad imputada, `Master`→`Mr`, `X` en cabina, `Person` sin edad).
3. **Confusiones detectadas** (por ejemplo, puerto vs. clase) y cómo las separaste.
4. **Qué NO podemos afirmar** (asociación ≠ causalidad; un solo naufragio; datos incompletos).

---

## Parte 10 — Del análisis a la predicción: formulación del problema (Pasos 39–42)

> **Teoría — explicar vs. predecir.** El EDA responde "¿qué se asocia con sobrevivir?". Un modelo responde "dado un pasajero nuevo, ¿qué probabilidad tiene de sobrevivir?". Para que esa promesa valga, hay que fijar **antes de entrenar** cuatro cosas: la **variable objetivo**, el **momento de predicción** (qué información existe *entonces*), la **métrica** y el **baseline** que el modelo debe superar. Si no se definen primero, es fácil elegir la métrica que mejor le quede al modelo que ya se entrenó.

**Dónde:** `notebooks/04_modelado.ipynb` (sección "10. Formulación") y `reports/problema_ml.md`. Celda de preparación del cuaderno (ejemplo resuelto):

```python
import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Los módulos con nombre numérico se importan con importlib (ver sección 3)
ml = importlib.import_module("flows.05_ml_models")

sns.set_theme(style="whitegrid", context="notebook")
X, y = ml.load_dataset()          # lee Silver; X con columnas candidatas, y = Survived
print(X.shape, y.mean())          # tasa de supervivencia ≈ 0.38
```

Y el esqueleto inicial de `flows/05_ml_models.py` (lo irás completando; las funciones llevan su Parte entre paréntesis):

```python
"""Modelado: Silver -> modelo entrenado y guardado."""
from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
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
CABIN_NUM = ["HasCabin"]            # features "sospechosas": se evalúan en el Paso 61
CABIN_CAT = ["TCabin"]
CANDIDATE_COLUMNS = NUM_FEATURES + CAT_FEATURES + CABIN_NUM + CABIN_CAT
PRIMARY_METRIC = "roc_auc"
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def load_dataset(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Lee Silver. Usa la edad ORIGINAL (con NaN): el modelo imputa por su cuenta."""
    df = pd.read_parquet(Path(path) if path else SILVER / SILVER_NAME)
    X = df[CANDIDATE_COLUMNS].copy()
    X["Age"] = df["AgeOriginal"]
    return X, df[TARGET].astype(int)
```

> **Por qué `load_dataset` reemplaza `Age` por `AgeOriginal`.** La `Age` de Silver ya fue imputada con estadísticas de *todos* los pasajeros. Usarla en el modelo filtraría información del conjunto de prueba (Parte 5, Principio 2.6).

### Paso 39. Definir el problema y la métrica

Crea `reports/problema_ml.md` completando esta tabla (cada fila con una justificación de una frase):

| Decisión | Tu respuesta |
|---|---|
| Tipo de problema | Clasificación binaria supervisada |
| Variable objetivo | `Survived` (1 = sobrevivió) |
| Unidad de análisis y momento de predicción | Un pasajero, *al embarcar* |
| Métrica principal | ¿ROC-AUC, F1, exactitud? ¿Por qué? |
| Métricas secundarias | precisión, recall, exactitud |
| Costo de errores | ¿Es peor predecir "sobrevive" a quien no sobrevivió, o al revés? Para una aseguradora, ¿qué significa cada error? |
| Criterio de éxito | Superar el baseline por un margen que puedas defender |

> **Teoría — métricas de clasificación.** Con ~38 % de positivos, un modelo que siempre dice "no sobrevive" acierta ~62 %: la **exactitud sola engaña**. *Precisión* = de los que predije positivos, ¿cuántos lo eran? *Recall* = de los positivos reales, ¿cuántos encontré? *F1* = media armónica de ambas. *ROC-AUC* = probabilidad de que el modelo asigne mayor puntaje a un positivo aleatorio que a un negativo aleatorio (no depende del umbral).

### Paso 40. Auditoría de variables y fuga de datos

Completa en el cuaderno una tabla con **todas** las columnas de Silver y decide si entran al modelo:

| Columna | ¿Existe antes del evento? | ¿Riesgo de fuga? | Decisión |
|---|---|---|---|
| `Survived`, `Survivor` | No (es el objetivo) | **Fuga directa** | Excluir |
| `PassengerId`, `Ticket`, `Name` | Sí | Identificadores: memorizan, no generalizan | Excluir (el título sí, vía `TTitulo`) |
| `NSex`, `NEmbarked`, `NCabin`, `NTitulo` | — | Duplican a `Sex`, etc. con un orden arbitrario | Excluir (usar *one-hot*) |
| `Age` imputada | — | Estadística calculada con todos los datos | Usar `AgeOriginal` |
| `Cabin`, `TCabin`, `HasCabin` | **Dudoso** | ¿Se registró la cabina *después* del naufragio (más fácil para los sobrevivientes)? | Evaluar con y sin ella (Paso 61) |
| `Person` | Derivada de `Age`/`Sex` | Redundante con ambas | Excluir del modelo base |

> **Pregunta de debate.** Si `HasCabin` mejora la métrica, ¿significa que tener cabina "protege"? ¿O que la información de cabina se conservó más para quienes sobrevivieron para contarla? Un modelo que explota un **artefacto del registro** no sirve para estimar riesgo real.

### Paso 41. División estratificada entrenamiento / prueba

> **Teoría — conjunto de prueba.** Es una evaluación **final** y se abre **una sola vez** (Parte 16). Todas las decisiones (modelo, hiperparámetros, umbral, variables) se toman con **validación cruzada sobre el entrenamiento**. *Estratificar* mantiene la proporción de sobrevivientes en ambos conjuntos. La **semilla** (`random_state`) hace la división reproducible.

**Dónde:** `flows/05_ml_models.py` → `split(X, y, test_size, random_state)`.
```python
def split(X: pd.DataFrame, y: pd.Series, test_size: float | None = None,
          random_state: int | None = None):
    """Devuelve X_train, X_test, y_train, y_test, estratificado por y."""
    # TODO: train_test_split(..., stratify=y); usa settings.* si los parámetros son None
    raise NotImplementedError
```

En el cuaderno: divide, verifica que la tasa de supervivencia es similar en ambos conjuntos (`y_train.mean()`, `y_test.mean()`) y **no vuelvas a tocar `X_test`** hasta la Parte 16.

### Paso 42. Baselines: lo mínimo que un modelo debe superar

**Lógica:** un modelo complejo que no supera a una regla simple no aporta valor.

```python
# Cuaderno, evaluando con validación cruzada sobre X_train/y_train (Paso 48 explica el método):
# 1. Baseline "ingenuo": DummyClassifier(strategy='most_frequent')  -> exactitud ≈ 0.62
# 2. Baseline "regla de género": predice 1 si Sex == 'female' y 0 en caso contrario.
#    Calcula exactitud, precisión, recall y F1 con sklearn.metrics.
#    Verificación: debería rondar ≈ 0.79 de exactitud. ¡Esa es la vara que hay que superar!
# 3. Registra los valores en reports/problema_ml.md como "criterio de éxito".
```

---

## Parte 11 — Preprocesamiento reproducible con `Pipeline` (Pasos 43–46)

> **Teoría — `Pipeline` y `ColumnTransformer`.** Un `Pipeline` encadena pasos (imputar → codificar → modelar) en **un solo objeto** con `fit` y `predict`. Al hacer `fit`, cada paso aprende **solo** de los datos de entrenamiento (medianas, categorías, medias); al hacer `predict`, aplica lo aprendido. Un `ColumnTransformer` aplica transformaciones distintas a columnas numéricas y categóricas. Resultado: el mismo preprocesamiento en entrenamiento, validación, producción y en el modelo guardado, sin copiar código.

### Paso 43. Demostrar la fuga de datos

**Dónde:** cuaderno 04.
```python
# Experimento (con la DESCRIPCIÓN, no hace falta código de modelo todavía):
# 1. Calcula la edad media de mujeres usando TODOS los datos (X) y usando solo X_train.
# 2. ¿Cuánto difieren? ¿Qué información de X_test "viajó" al entrenamiento en el primer caso?
# 3. Escribe en 3 líneas por qué la diferencia es pequeña aquí y por qué igualmente es un error
#    de método (en otros datasets, con más faltantes o grupos pequeños, puede ser grande).
```

### Paso 44. Un transformador propio: imputación de edad por grupo (ejemplo resuelto)

El promedio por género del Paso 22, ahora como **transformador de scikit-learn**: `fit` aprende los promedios con datos de entrenamiento y `transform` los aplica a cualquier dato. Si aparece un grupo no visto, usa el promedio global de entrenamiento.

Añade a `flows/05_ml_models.py`:

```python
class GroupAgeImputer(BaseEstimator, TransformerMixin):
    """Rellena 'Age' con el promedio de su grupo, aprendido SOLO en fit()."""

    def __init__(self, group_cols=("Sex",), target="Age"):
        self.group_cols = group_cols
        self.target = target

    def fit(self, X, y=None):
        cols = list(self.group_cols)
        self.group_means_ = X.groupby(cols)[self.target].mean()
        self.global_mean_ = X[self.target].mean()
        return self

    def transform(self, X):
        X = X.copy()
        cols = list(self.group_cols)
        means = self.group_means_.rename("_fill").reset_index()
        fill = X[cols].merge(means, on=cols, how="left")["_fill"].to_numpy(dtype=float)
        fill = np.where(np.isnan(fill), self.global_mean_, fill)
        X[self.target] = X[self.target].fillna(pd.Series(fill, index=X.index))
        return X
```

### Paso 45. Construir el preprocesador y el pipeline completo

**Lógica:** numéricas → mediana (por si quedara algún faltante, p. ej. `Fare`) + estandarización; categóricas → imputar con la más frecuente + *one-hot*. `handle_unknown="infrequent_if_exist"` con `min_frequency` agrupa las categorías muy raras (como una cubierta con un solo pasajero) en lugar de crear columnas casi vacías.

**Dónde:** `build_pipeline(model, num_features, cat_features)`.
```python
def build_pipeline(model, num_features=None, cat_features=None) -> Pipeline:
    """GroupAgeImputer -> ColumnTransformer -> modelo, todo en un Pipeline."""
    num_features = NUM_FEATURES if num_features is None else num_features
    cat_features = CAT_FEATURES if cat_features is None else cat_features
    # 1. num = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    # 2. cat = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
    #                    ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=10))])
    # 3. prep = ColumnTransformer([('num', num, num_features), ('cat', cat, cat_features)],
    #                             sparse_threshold=0)       # salida densa: todos los modelos la aceptan
    # 4. return Pipeline([('age_imputer', GroupAgeImputer(group_cols=('Sex', 'Pclass'))),
    #                     ('prep', prep), ('model', model)])
    # Nota: si quitas 'Age' de num_features, GroupAgeImputer sigue funcionando (ignora lo que no use)
    #       pero 'Sex' y 'Pclass' deben seguir presentes en X.
    raise NotImplementedError
```

> **Decisión que debes justificar.** Aquí la imputación agrupa por `Sex` **y** `Pclass` (más específica que la del Paso 22). Compara en el cuaderno ambas variantes con validación cruzada (Paso 48) y anota cuál prefieres y por qué.

### Paso 46. Verificar el preprocesamiento

**Dónde:** cuaderno y `tests/test_05_ml_models.py`.
```python
# Cuaderno:
# 1. pipe = ml.build_pipeline(DummyClassifier()); pre = pipe[:-1].fit(X_train)
# 2. Z = pre.transform(X_test): ¿tiene NaN? (np.isnan(Z).any())  ¿cuántas columnas?
# 3. Nombres de columnas resultantes: pre['prep'].get_feature_names_out()
# 4. Comprueba que 'Sex' quedó como 'cat__Sex_female' / 'cat__Sex_male' (sin orden implícito).
```

Crea `tests/test_05_ml_models.py` con estas pruebas (las de las Partes 12–17 se añaden más adelante):

```python
import importlib

import numpy as np
import pandas as pd
import pytest

ml = importlib.import_module("flows.05_ml_models")


@pytest.fixture
def toy():
    """Dataset sintético con las columnas candidatas, para probar sin depender de Silver."""
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame({
        "Age": np.where(rng.random(n) < 0.2, np.nan, rng.uniform(1, 70, n)),
        "SibSp": rng.integers(0, 4, n),
        "Parch": rng.integers(0, 3, n),
        "Fare": rng.uniform(5, 100, n),
        "Familia": rng.integers(0, 6, n),
        "Pclass": rng.choice([1, 2, 3], n),
        "Sex": rng.choice(["male", "female"], n),
        "Embarked": rng.choice(["S", "C", "Q"], n),
        "TTitulo": rng.choice(["Mr", "Mrs", "Miss", "Otro"], n),
        "Alone": rng.choice(["Alone", "with Family"], n),
        "HasCabin": rng.integers(0, 2, n),
        "TCabin": rng.choice(list("XXXXCB"), n),
    })
    y = pd.Series(((X["Sex"] == "female") & (rng.random(n) < 0.9)).astype(int))
    return X, y


def test_group_age_imputer_uses_only_training_statistics():
    train = pd.DataFrame({"Sex": ["male", "male", "female"], "Age": [10.0, 20.0, 40.0]})
    test = pd.DataFrame({"Sex": ["male", "other"], "Age": [np.nan, np.nan]})
    out = ml.GroupAgeImputer(group_cols=("Sex",)).fit(train).transform(test)
    assert out.loc[0, "Age"] == 15.0                          # promedio de hombres en TRAIN
    assert out.loc[1, "Age"] == pytest.approx(70 / 3)         # grupo no visto: promedio global de TRAIN


def test_pipeline_fits_and_predicts_probabilities(toy):
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (len(X),) and ((0 <= proba) & (proba <= 1)).all()


def test_split_is_stratified_and_reproducible(toy):
    X, y = toy
    a = ml.split(X, y, test_size=0.25, random_state=1)
    b = ml.split(X, y, test_size=0.25, random_state=1)
    pd.testing.assert_index_equal(a[0].index, b[0].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.03


def test_candidates_include_baseline_and_are_seeded():
    cands = ml.get_candidates(42)
    assert "dummy" in cands and "logreg" in cands and len(cands) >= 5
```

---

## Parte 12 — Modelos candidatos y validación cruzada (Pasos 47–50)

> **Teoría — sobreajuste y validación cruzada.** Un modelo puede memorizar el entrenamiento y fallar con datos nuevos (*sobreajuste*). La **validación cruzada estratificada de k pliegues** divide el entrenamiento en *k* partes: entrena con *k−1* y evalúa en la restante, rotando. Obtienes *k* puntajes: su **media** estima el desempeño y su **desviación** indica la estabilidad. Con 891 pasajeros, la varianza es grande: no trates diferencias de 0.5 puntos como reales.

### Paso 47. Definir los modelos candidatos

**Lógica:** una familia variada — un baseline, un modelo lineal interpretable, un árbol simple, y dos ensambles — cada uno con la **semilla** fija.

**Dónde:** `get_candidates(random_state)`.
```python
def get_candidates(random_state: int = 42) -> dict:
    """Diccionario {nombre: estimador sin entrenar}."""
    # 'dummy'  : DummyClassifier(strategy='most_frequent')
    # 'logreg' : LogisticRegression(max_iter=1000)
    # 'tree'   : DecisionTreeClassifier(max_depth=4, random_state=random_state)
    # 'forest' : RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    # 'hgb'    : HistGradientBoostingClassifier(random_state=random_state)
    # 'knn'    : KNeighborsClassifier()
    raise NotImplementedError
```

### Paso 48. Validación cruzada de todos los candidatos

**Dónde:** `cross_validate_candidates(X_train, y_train, ...)` y cuaderno.
```python
def cross_validate_candidates(X, y, num_features=None, cat_features=None,
                              random_state=None, cv_folds=None) -> pd.DataFrame:
    """Devuelve un DataFrame con media y desviación de cada métrica por modelo."""
    # 1. cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    # 2. Para cada (nombre, modelo): pipe = build_pipeline(modelo, ...);
    #    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, return_train_score=True)
    # 3. Agrega por modelo: res['test_roc_auc'].mean(), .std(), res['train_roc_auc'].mean(), etc.
    # 4. Registra con log.info y devuelve la tabla ordenada por la métrica principal.
    raise NotImplementedError
```

### Paso 49. Leer los resultados

**Dónde:** cuaderno.
```python
# 1. Muestra la tabla de resultados (media ± desviación por métrica y modelo).
# 2. Gráfico: boxplot de los puntajes por pliegue de la métrica principal para cada modelo.
#    (guarda los puntajes por pliegue en una lista de dicts y conviértelos a DataFrame largo).
# 3. Calcula la BRECHA ENTRENAMIENTO − VALIDACIÓN de cada modelo.
#    ¿Cuál sobreajusta más? ¿Qué relación tiene con su flexibilidad?
```

### Paso 50. Elegir finalistas con criterio

**Dónde:** `reports/problema_ml.md`, sección "Selección de modelos".

Elige **2 o 3 finalistas** justificando, por modelo: desempeño medio, estabilidad (desviación), brecha entrenamiento-validación, interpretabilidad y costo computacional. Un modelo ligeramente peor pero estable y explicable puede ser mejor elección que el primero de la tabla.

> **Regla de oro.** Nunca elijas un modelo mirando el conjunto de prueba.

---

## Parte 13 — Evaluación en profundidad (Pasos 51–55)

> **Teoría.** Un solo número oculta qué errores comete el modelo. Estudiaremos las **predicciones fuera de muestra** del entrenamiento (cada pasajero se predice con un modelo que *no lo vio*), lo que permite mirar matrices de confusión, curvas y errores por subgrupo sin tocar el conjunto de prueba.

### Paso 51. Predicciones fuera de muestra y matriz de confusión

**Dónde:** cuaderno.
```python
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

# 1. cv = StratifiedKFold(5, shuffle=True, random_state=ml.settings.random_state)
# 2. proba = cross_val_predict(pipe_finalista, X_train, y_train, cv=cv, method='predict_proba')[:, 1]
# 3. pred = (proba >= 0.5).astype(int)
# 4. ConfusionMatrixDisplay.from_predictions(y_train, pred) y classification_report(y_train, pred)
# 5. Explica con tus palabras qué significa cada celda para el problema del seguro.
```

### Paso 52. Curvas ROC y de precisión-recall

```python
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# 1. RocCurveDisplay.from_predictions(y_train, proba) para cada finalista (en un mismo eje).
# 2. PrecisionRecallDisplay.from_predictions(y_train, proba).
# 3. ¿Cuándo es más informativa la curva PR que la ROC? (pista: clases desbalanceadas)
```

### Paso 53. Calibración y elección del umbral

> **Teoría.** `predict_proba` devuelve un puntaje; "0.5" es solo una convención. Un modelo está **calibrado** si, entre los pasajeros con probabilidad 0.7, sobrevivió cerca del 70 %. Y el **umbral** debe elegirse según el costo de cada error, **con datos de validación**, no con el conjunto de prueba.

```python
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_recall_curve

# 1. prob_true, prob_pred = calibration_curve(y_train, proba, n_bins=8); grafícala vs. la diagonal.
# 2. brier_score_loss(y_train, proba): ¿qué tan lejos de 0 (perfecto) y de 0.25 (azar)?
# 3. Con precision_recall_curve, encuentra el umbral que maximiza F1 en las predicciones
#    fuera de muestra y compáralo con 0.5.
# 4. Decide un umbral y regístralo (lo usarás en la Parte 17: THRESHOLD).
```

### Paso 54. Análisis de errores por subgrupos

```python
# 1. Construye un DataFrame con X_train, y_train, pred y un indicador de error (pred != y_train).
# 2. Calcula la exactitud y el recall por Sex, por Pclass y por Person (o por edad agrupada).
# 3. ¿Hay subgrupos donde el modelo es mucho peor? ¿Con cuántos pasajeros (n)?
# 4. Examina 10 errores concretos de alta confianza (proba muy alta o muy baja y pred equivocada):
#    ¿qué tienen en común? ¿Qué variable podría ayudar?
```

> **Reflexión ética.** El modelo usa `Sex` y `Pclass`, que dominaron las decisiones de evacuación de 1912. Predecir bien *lo que ocurrió* no implica que *deba* ocurrir así ni que esas variables sean apropiadas para decidir quién recibe un servicio hoy. Anota en `reports/problema_ml.md` en qué contextos usarías (o no) este modelo.

### Paso 55. Curva de aprendizaje: ¿más datos o un modelo distinto?

```python
from sklearn.model_selection import learning_curve

# 1. sizes, tr, va = learning_curve(pipe_finalista, X_train, y_train, cv=cv,
#                                   train_sizes=np.linspace(0.2, 1.0, 5), scoring='roc_auc')
# 2. Grafica la media (y la banda de ±1 desviación) de tr y va contra sizes.
# 3. Diagnóstico: ¿las curvas convergen en un valor bajo (alto sesgo: modelo demasiado simple /
#    variables insuficientes) o hay una brecha grande (alta varianza: más datos o más regularización)?
```

---

## Parte 14 — Ajuste de hiperparámetros (Pasos 56–58)

> **Teoría — hiperparámetros.** Son los "diales" del modelo que no se aprenden de los datos (profundidad del árbol, regularización `C`, tasa de aprendizaje). Se buscan **dentro de la validación cruzada del entrenamiento**. Cuantas más combinaciones pruebes, más probable es que alguna gane *por azar*: el puntaje del mejor candidato es **optimista**. Por eso la evaluación honesta se hace después, sobre datos que la búsqueda nunca vio.

### Paso 56. Definir el espacio de búsqueda

**Dónde:** `get_param_grids()`.
```python
def get_param_grids() -> dict[str, dict]:
    """Rejillas por modelo, con los nombres que espera el Pipeline (prefijo 'model__')."""
    # 'logreg': {'model__C': [0.01, 0.1, 1, 10], 'model__penalty': ['l2']}
    # 'forest': {'model__n_estimators': [200, 400], 'model__max_depth': [4, 6, 8, None],
    #            'model__min_samples_leaf': [1, 3, 5]}
    # 'hgb'   : {'model__learning_rate': [0.03, 0.1], 'model__max_depth': [2, 3, 4],
    #            'model__max_iter': [100, 200], 'model__l2_regularization': [0.0, 1.0]}
    raise NotImplementedError
```

### Paso 57. Búsqueda con validación cruzada

```python
from sklearn.model_selection import GridSearchCV

# En el cuaderno, para cada finalista:
# 1. gs = GridSearchCV(ml.build_pipeline(modelo), grid, scoring=ml.PRIMARY_METRIC, cv=cv,
#                      n_jobs=-1, refit=True)
# 2. gs.fit(X_train, y_train)      # SOLO entrenamiento
# 3. gs.best_params_, gs.best_score_ y pd.DataFrame(gs.cv_results_) ordenado por rank_test_score
# 4. ¿Qué tan plana es la superficie? Si los 5 mejores tienen puntajes casi iguales, el ajuste
#    importa poco y conviene elegir el más simple.
```

### Paso 58. ¿Mejoró de verdad? (CV anidada o comparación pareada)

**Lógica:** `best_score_` es optimista. Dos formas de comprobar si el ajuste aportó algo:

```python
# Opción A — CV anidada: cross_val_score(GridSearchCV(...), X_train, y_train, cv=cv_externo)
#            estima el desempeño DEL PROCEDIMIENTO COMPLETO "buscar + entrenar".
# Opción B — comparación pareada: usa los MISMOS pliegues para el modelo por defecto y el ajustado
#            y observa la diferencia pliegue a pliegue (media y desviación de las diferencias).
# Concluye en una frase: ¿la mejora supera la variabilidad entre pliegues?
```

---

## Parte 15 — Interpretación y ablación de variables (Pasos 59–61)

> **Teoría — importancia ≠ causalidad.** Un modelo predictivo mide qué variables le sirven para *predecir*, no qué *causa* sobrevivir. Y las explicaciones pueden diferir según el método: contrástalas.

### Paso 59. Coeficientes del modelo lineal

```python
# 1. Ajusta el pipeline con LogisticRegression sobre X_train.
# 2. nombres = pipe['prep'].get_feature_names_out(); coefs = pipe['model'].coef_[0]
# 3. Tabla ordenada por |coef| y razones de momios (odds ratios) = np.exp(coef).
# 4. Interpreta 3 coeficientes con un enunciado claro (ojo: las numéricas están estandarizadas).
```

### Paso 60. Importancia por permutación

> **Teoría.** Se baraja una columna en datos **no vistos** y se mide cuánto cae la métrica; cuanto más cae, más dependía el modelo de ella. Es preferible a la importancia por impurezas de los árboles, que sobrevalora variables con muchos valores distintos.

```python
from sklearn.inspection import permutation_importance

# 1. X_tr, X_val, y_tr, y_val = ml.split(X_train, y_train, test_size=0.25)   # NO uses X_test
# 2. Ajusta el finalista con (X_tr, y_tr).
# 3. r = permutation_importance(pipe, X_val, y_val, scoring='roc_auc', n_repeats=30, random_state=42)
#    (se permutan las COLUMNAS ORIGINALES, no las generadas por one-hot)
# 4. Gráfico de barras con media ± desviación. Compara el orden con la correlación del Paso 27
#    y con las tasas de la Parte 9. ¿Coinciden? ¿Qué variable cambia de lugar y por qué?
```

### Paso 61. Ablación: ¿cuánto aporta cada grupo de variables?

**Lógica:** reentrena con y sin un grupo de variables, **con los mismos pliegues**, y mide el cambio.

```python
# Cuaderno, usando ml.cross_validate_candidates o cross_validate directamente con:
#   A) base (NUM_FEATURES + CAT_FEATURES)
#   B) base + cabina (HasCabin, TCabin)       <- recuerda la duda del Paso 40
#   C) base sin TTitulo
#   D) base sin Sex (¿qué variables "toman el relevo"?)
# Tabla: configuración | ROC-AUC (media ± desv.) | ΔAUC vs. A.
# Decisión escrita: ¿incluirás las variables de cabina en el modelo final? Justifica con
# desempeño Y con la discusión del artefacto de registro.
```

---

## Parte 16 — Evaluación final sobre el conjunto de prueba (Pasos 62–63)

> **Teoría — abrir el conjunto de prueba una sola vez.** Todo lo anterior se decidió con el entrenamiento. Ahora se entrena el modelo **elegido** (con sus hiperparámetros y su umbral) con *todo* `X_train` y se evalúa **una vez** en `X_test`. Si el resultado te decepciona, **no** vuelvas atrás a "arreglarlo" mirando el conjunto de prueba: documenta el resultado tal cual.

### Paso 62. Evaluar en prueba con incertidumbre

**Dónde:** `evaluate_on_test(pipeline, X_test, y_test, threshold)` y cuaderno.
```python
def evaluate_on_test(pipeline, X_test, y_test, threshold: float = 0.5) -> dict:
    """Devuelve exactitud, F1, ROC-AUC y el tamaño de la prueba (dict serializable a JSON)."""
    # TODO: proba = pipeline.predict_proba(X_test)[:, 1]; pred = (proba >= threshold).astype(int)
    raise NotImplementedError
```
```python
# Cuaderno:
# 1. Ajusta el finalista con (X_train, y_train) y llama a evaluate_on_test UNA vez.
# 2. Intervalo de confianza por bootstrap: remuestrea (con reemplazo) los pares (y_test, proba)
#    1000 veces, calcula la métrica en cada remuestreo y toma los percentiles 2.5 y 97.5.
# 3. Con ~180 pasajeros de prueba, ¿qué tan ancho es el intervalo? ¿Qué implica para comparar modelos?
```

### Paso 63. Comparar, contrastar y documentar limitaciones

```python
# 1. Tabla final: baseline de género vs. dummy vs. modelo elegido, con CV (entrenamiento) y prueba.
# 2. ¿La estimación de prueba es coherente con la de validación cruzada? Si difieren mucho, ¿por qué?
# 3. Completa en reports/problema_ml.md la sección "Limitaciones": tamaño de muestra, un solo
#    suceso histórico, edad/cabina faltantes, intervalo de confianza, variables sensibles (Sex, Pclass)
#    y qué decisiones NO debería respaldar este modelo.
```

Añade a `tests/test_05_ml_models.py`:
```python
def test_evaluate_on_test_returns_metrics(toy):
    X, y = toy
    X_tr, X_te, y_tr, y_te = ml.split(X, y, test_size=0.25, random_state=1)
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X_tr, y_tr)
    res = ml.evaluate_on_test(pipe, X_te, y_te, threshold=0.5)
    assert {"accuracy", "f1", "roc_auc", "n_test"} <= set(res)
    assert 0 <= res["accuracy"] <= 1 and res["n_test"] == len(X_te)
```

---

## Parte 17 — Entrenamiento final y guardado del modelo (Pasos 64–69)

> **Teoría — del experimento al artefacto.** Terminada la investigación, se **congelan las decisiones** y se entrena el modelo definitivo con **todos** los datos etiquetados disponibles (entrenamiento + prueba): más datos suelen dar un modelo algo mejor, y su desempeño esperado es el que estimaste en la Parte 16 (el procedimiento ya está validado). El modelo se guarda **junto con la información que permite auditarlo**: qué datos vio, con qué parámetros, qué versiones de librerías, qué métricas obtuvo.
>
> **Seguridad.** Los archivos `.joblib` (como `pickle`) pueden ejecutar código al cargarse: **nunca cargues un modelo de una fuente que no controles**, y recuerda que un modelo guardado con una versión de scikit-learn puede no abrirse o comportarse distinto con otra: por eso se registran las versiones.

### Paso 64. Congelar las decisiones en el módulo

Las decisiones que tomaste en el cuaderno 04 se registran **en el código, bajo revisión**, no en la memoria de quien hizo el experimento. Añade a `flows/05_ml_models.py` (con tus valores):

```python
# --- Decisiones congeladas (resultado del cuaderno 04; justificadas en reports/problema_ml.md) ---
MODEL_NAME = "titanic_model_v1"
FINAL_ESTIMATOR = "hgb"                 # nombre en get_candidates()
FINAL_PARAMS = {                        # mejores hiperparámetros del Paso 57 (ejemplo; usa los tuyos)
    "model__learning_rate": 0.05,
    "model__max_depth": 3,
    "model__max_iter": 200,
    "model__l2_regularization": 1.0,
}
USE_CABIN_FEATURES = False              # decisión del Paso 61
THRESHOLD = 0.5                         # decisión del Paso 53
```

### Paso 65. Entrenar el modelo final con todos los datos etiquetados

**Dónde:** `train_final(X, y, ...)`.
```python
def train_final(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Construye el pipeline con las decisiones congeladas y lo entrena con TODO (X, y)."""
    # 1. num = NUM_FEATURES + (CABIN_NUM si USE_CABIN_FEATURES); cat = CAT_FEATURES + (CABIN_CAT si ...)
    # 2. modelo = get_candidates(settings.random_state)[FINAL_ESTIMATOR]
    # 3. pipe = build_pipeline(modelo, num, cat).set_params(**FINAL_PARAMS)
    # 4. return pipe.fit(X, y)
    raise NotImplementedError
```

### Paso 66. Guardar el modelo con sus metadatos (ejemplo resuelto)

Añade a `flows/05_ml_models.py`:

```python
def save_model(pipeline: Pipeline, metadata: dict, name: str = MODEL_NAME,
               models_dir: Path = MODELS) -> Path:
    """Guarda el pipeline (.joblib) y un archivo de metadatos (.metadata.json)."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{name}.joblib"
    joblib.dump(pipeline, model_path)
    meta = {
        **metadata,
        "name": name,
        "saved_at": datetime.now(timezone.utc).isoformat(),
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
    """Carga el pipeline y sus metadatos; avisa si la versión de scikit-learn cambió."""
    path = Path(path)
    pipeline = joblib.load(path)
    meta = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if meta.get("sklearn_version") != sklearn.__version__:
        log.warning("Modelo entrenado con scikit-learn %s; versión actual %s",
                    meta.get("sklearn_version"), sklearn.__version__)
    return pipeline, meta
```

Los **metadatos** que debes pasar desde `run()` (mínimo): `estimator`, `params` (`FINAL_PARAMS`), `features` (listas num. y cat.), `target`, `threshold`, `random_state`, `n_train` (filas con las que se entrenó), `data_sha256` (huella de `data/silver/titanic_silver.parquet` con `sha256_of`) y `metrics` (las métricas de **prueba** de la Parte 16, con su intervalo).

### Paso 67. Cargar y predecir sobre pasajeros nuevos

```python
def predict(pipeline: Pipeline, X: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """Devuelve un DataFrame con 'proba_survive' y 'pred' (0/1) para cada fila de X."""
    # TODO: usa pipeline.predict_proba(X)[:, 1] y el umbral recibido.
    raise NotImplementedError
```

Pruebas de ida y vuelta (añade a `tests/test_05_ml_models.py`):
```python
def test_save_and_load_roundtrip(tmp_path, toy):
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    path = ml.save_model(pipe, {"estimator": "logreg", "n_train": len(X)},
                         name="toy_model", models_dir=tmp_path)
    loaded, meta = ml.load_model(path)
    assert np.array_equal(pipe.predict(X), loaded.predict(X))     # mismas predicciones
    assert {"sklearn_version", "python_version", "model_sha256", "saved_at"} <= set(meta)
    assert meta["n_train"] == len(X)


def test_predict_returns_probabilities_and_labels(toy):
    X, y = toy
    pipe = ml.build_pipeline(ml.get_candidates(42)["logreg"]).fit(X, y)
    out = ml.predict(pipe, X, threshold=0.5)
    assert list(out.columns) == ["proba_survive", "pred"]
    assert set(out["pred"].unique()) <= {0, 1}
```

### Paso 68. Orquestar todo: `run()` y `make pipeline`

**Dónde:** `run()` de `flows/05_ml_models.py`; el orden lo da `job.py`.
```python
def run(in_dir: Path | None = None, models_dir: Path = MODELS) -> Path:
    """Silver -> evaluación honesta -> modelo final entrenado y guardado."""
    # 1. X, y = load_dataset(in_dir / SILVER_NAME si in_dir, o el valor por defecto)
    # 2. X_tr, X_te, y_tr, y_te = split(X, y)
    # 3. Evaluación: pipe_eval = train_final(X_tr, y_tr); metrics = evaluate_on_test(pipe_eval, X_te, y_te, THRESHOLD)
    #    log.info con las métricas (esta es la estimación que se reporta).
    # 4. Modelo definitivo: pipeline = train_final(X, y)    # con TODOS los datos etiquetados
    # 5. save_model(pipeline, metadata, models_dir=models_dir) con los metadatos del Paso 66.
    # 6. Devuelve la ruta del modelo.
    raise NotImplementedError
```

Verificación de **extremo a extremo desde cero** (lo que haría quien revise tu trabajo):
```bash
git clone <tu-repositorio> && cd template_dsai
python -m venv .venv && source .venv/bin/activate
make install
make test         # todas las pruebas en verde
make pipeline     # ingesta -> silver -> gold -> modelo guardado
ls data/bronze data/silver data/gold models
```

### Paso 69. Prueba de humo con pasajeros sin etiqueta y ficha del modelo

**Prueba de humo.** Comprueba que el modelo guardado funciona sobre datos que nunca vio y que **no tienen** la respuesta (Bronze sin etiqueta). No hay métrica que calcular: el objetivo es verificar que el flujo de carga → preprocesamiento → predicción corre de punta a punta. La evaluación sobre esos pasajeros se hará en el Taller #3.

```python
# Cuaderno 04, última sección:
from flows.common.paths import BRONZE, MODELS, BRONZE_UNLABELED_NAME
pipe, meta = ml.load_model(MODELS / "titanic_model_v1.joblib")
nuevos = pd.read_csv(BRONZE / BRONZE_UNLABELED_NAME)
# Aplica las MISMAS transformaciones deterministas de Silver (build_silver) a 'nuevos',
# toma ml.CANDIDATE_COLUMNS y usa 'AgeOriginal' como 'Age', igual que load_dataset().
# Predice con ml.predict(pipe, X_nuevos) y revisa: ¿hay NaN? ¿la distribución de probabilidades
# es razonable? ¿alguna categoría nueva (p. ej. un título) que el modelo no conoce?
```

**Ficha del modelo (`reports/model_card.md`).** Documento breve (una página) con: propósito y uso previsto, datos de entrenamiento (fuente, huella SHA-256, tamaño, periodo), variables usadas y excluidas, métricas de validación y de prueba **con su incertidumbre**, desempeño por subgrupos (Paso 54), limitaciones y usos no recomendados (Paso 63), versiones de Python/librerías y cómo reproducir (`make pipeline`).

### Definición de terminado (lista de verificación final)

- [ ] `make test` pasa completo y `make lint` no reporta errores.
- [ ] `make pipeline` genera `data/bronze`, `data/silver`, `data/gold` y `models/titanic_model_v1.joblib` + `.metadata.json` desde un clon limpio.
- [ ] No hay rutas absolutas, credenciales ni datos/modelos en Git (`git status` limpio de `data/`, `models/`).
- [ ] Los cuadernos corren de arriba abajo (*Restart & Run All*) y no contienen lógica productiva.
- [ ] `reports/conclusiones_eda.md`, `reports/problema_ml.md` y `reports/model_card.md` están completos y distinguen asociación de causalidad.
- [ ] El conjunto de prueba se usó **una sola vez** y las métricas finales incluyen su incertidumbre.
- [ ] El historial de *commits* cuenta la historia del proyecto (un cambio por *commit*, mensajes que explican el porqué).

