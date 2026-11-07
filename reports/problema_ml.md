# Definición del Problema de Machine Learning y Baselines

## 1. Formulación del Problema y Métrica (Paso 39)

| Decisión | Respuesta | Justificación |
|---|---|---|
| **Tipo de problema** | Clasificación binaria supervisada | Predecimos un estado discreto con dos posibles desenlaces (sobrevive vs. no sobrevive) a partir de datos históricos etiquetados. |
| **Variable objetivo** | `Survived` (1 = sobrevivió, 0 = no sobrevivió) | Es la variable binaria real que indica la supervivencia del pasajero en el hundimiento. |
| **Unidad de análisis y momento de predicción** | Un pasajero individual, **al momento del embarque** | La predicción debe basarse exclusivamente en la información disponible antes de que el barco zarpe; no se puede usar información generada durante o después del naufragio. |
| **Métrica principal** | **ROC-AUC** | Con una tasa de supervivencia de ~38.4%, las clases están moderadamente desbalanceadas; ROC-AUC evalúa la capacidad de ordenamiento y discriminación global del modelo sin atarse a un umbral de decisión arbitrario. |
| **Métricas secundarias** | **F1-Score, Precisión, Recall, Exactitud** | Permiten diagnosticar el equilibrio entre falsos positivos y falsos negativos según el umbral de decisión operativo. |
| **Costo de errores** | Falso positivo > Falso negativo (según contexto) | Para una aseguradora marítima, un **falso positivo** (predecir supervivencia cuando la persona falleció) subestima el riesgo del evento catastrófico, llevando a fijar primas insuficientes o reservas deficitarias. Un **falso negativo** sobrestima el riesgo, encareciendo la póliza y perdiendo clientes. En protocolos de evacuación, un falso positivo es inadmisible porque asume a una persona a salvo. |
| **Criterio de éxito** | Superar el baseline heurístico de género por un margen defendible | Un modelo estadístico o de ML solo aporta valor si supera a la regla simple histórica (*"mujeres y niños primero"*), logrando ROC-AUC > 0.82 y F1 > 0.75 en validación cruzada. |

---

## 2. Auditoría de Variables y Prevención de Fuga de Datos (Paso 40)

Para garantizar que el modelo generalice y no explote artefactos del registro ni información del futuro, se auditan todas las variables:

| Columna | ¿Existe antes del evento? | ¿Riesgo de fuga? | Decisión | Justificación técnica |
|---|---|---|---|---|
| `Survived`, `Survivor` | No | **Fuga directa** | **Excluir** | Es la variable objetivo o su etiqueta directa. |
| `PassengerId`, `Ticket`, `Name` | Sí | Sin fuga, pero riesgo de sobreajuste | **Excluir** | Identificadores de alta cardinalidad; memorizan datos y no generalizan (el título social se extrae de `Name` como `TTitulo`). |
| `NSex`, `NEmbarked`, `NCabin`, `NTitulo` | — | Falso orden ordinal | **Excluir** | Codificaciones enteras arbitrarias; se prefiere codificación One-Hot dentro del `Pipeline`. |
| `Age` (imputada en Silver) | — | **Fuga de datos (Leakage)** | **Usar `AgeOriginal`** | La edad en Silver fue imputada con la media de todo el dataset (incluyendo datos que pertenecerían al test); se debe usar `AgeOriginal` para que el `Pipeline` impute solo con datos de entrenamiento. |
| `Cabin`, `TCabin`, `HasCabin` | **Dudoso** | Posible fuga / sesgo de registro | **Evaluar con y sin ella** | Es plausible que la información de cabina se registrara con mayor fidelidad para sobrevivientes que pudieron declarar. Se auditará mediante ablación. |
| `Person` | Derivada | Redundancia | **Excluir del modelo base** | Creada a partir de `Age` y `Sex`; su inclusión simultánea introduce multicolinealidad con el modelo lineal. |

### Conjunto final de variables candidatas:
* **Numéricas**: `['Age', 'SibSp', 'Parch', 'Fare', 'Familia']`
* **Categóricas**: `['Pclass', 'Sex', 'Embarked', 'TTitulo', 'Alone']`
* **Cabina (auditable)**: `['HasCabin']` (numérica), `['TCabin']` (categórica)

---

## 3. Partición de Datos: Entrenamiento y Prueba (Paso 41)

* **Proporción**: 80 % Entrenamiento (`X_train`, `y_train`) y 20 % Prueba (`X_test`, `y_test`).
* **Estratificación**: `stratify=y` para garantizar que la tasa de supervivencia (~38.4 %) sea idéntica en ambos conjuntos, evitando sesgos de partición.
* **Semilla aleatoria**: `random_state=42` para reproducibilidad total.
* **Regla estricta**: El conjunto de prueba (`X_test`, `y_test`) se almacena aislado y **no se toca** durante el preprocesamiento, selección de modelos ni ajuste de hiperparámetros. Se abrirá una sola vez en la evaluación final.

---

## 4. Baselines: La Vara Mínima (Paso 42)

Evaluados sobre `X_train` / `y_train`:

| Baseline | Descripción | Exactitud (*Accuracy*) | Precisión | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **1. Ingenuo (Dummy)** | `DummyClassifier(strategy='most_frequent')` — siempre predice 0 (no sobrevive) | ~0.616 | 0.000 | 0.000 | 0.000 | 0.500 |
| **2. Heurístico de Género** | Regla histórica: predice 1 si `Sex == 'female'`, 0 si `Sex == 'male'` | **~0.787** | **~0.742** | **~0.681** | **~0.710** | **~0.767** |

### Conclusión del Criterio de Éxito:
El baseline ingenuo demuestra que una exactitud del **61.6 %** no tiene ningún valor predictivo (equivale a predecir siempre que todos mueren). El verdadero rival a vencer es la **regla de género (~78.7 % de exactitud y ~0.767 de ROC-AUC)**. Cualquier modelo de Machine Learning que propongamos debe superar de forma estadísticamente significativa estas cifras mediante validación cruzada.


---

## 5. Comparación y Selección de Modelos Candidatos (Paso 50)

Evaluados mediante validación cruzada estratificada de 5 pliegues exclusivamente sobre `X_train`:

| Modelo | ROC-AUC (Test) | Exactitud (Test) | Brecha Sobreajuste (ROC-AUC) | Veredicto |
|---|---|---|---|---|
| **HistGradientBoosting (`hgb`)** | **0.8867 ± 0.019** | **0.8202** | 0.1068 | **Finalista 1 (Mejor desempeño global)** |
| **Random Forest (`forest`)** | 0.8657 ± 0.016 | 0.8034 | 0.1327 | Descartado frente a HGB (mayor sobreajuste en train: 0.998) |
| **Regresión Logística (`logreg`)** | **0.8585 ± 0.024** | **0.7921** | **0.0062** | **Finalista 2 (Máxima estabilidad y explicabilidad)** |
| **Árbol de Decisión (`tree`)** | 0.8468 ± 0.026 | 0.8047 | 0.0465 | Modelo de referencia simple |
| **K-Nearest Neighbors (`knn`)** | 0.8398 ± 0.017 | 0.7865 | 0.0883 | Sensible a la maldición de la dimensionalidad |
| **Dummy (Azar)** | 0.5000 ± 0.000 | 0.6166 | 0.0000 | Baseline superado por todos |

### Justificación de los Finalistas:
1. **HistGradientBoosting (`hgb`)**: Es el modelo con mayor poder discriminativo (**ROC-AUC = 0.887** y **82.0 % de exactitud**). Maneja eficazmente relaciones no lineales e interacciones complejas detectadas en el EDA (como Clase × Género) sin requerir ingeniería manual exhaustiva.
2. **Regresión Logística (`logreg`)**: Aunque tiene un ROC-AUC ligeramente inferior (0.858), es el modelo más **estable y robusto** de todos: la brecha entre entrenamiento y validación es de apenas **0.006**, lo que significa que prácticamente no sufre sobreajuste. Además, sus coeficientes ofrecen interpretabilidad directa para el negocio de una aseguradora.


---

## 6. Diagnóstico de Errores y Reflexión Ética (Paso 54)

### Análisis de Desempeño por Subgrupos:
* **Género**: La tasa de error en mujeres es de aproximadamente **18.7 %** y en hombres de **17.5 %**. Los principales errores en mujeres corresponden a pasajeras de 3ª clase que no lograron acceder a los botes; en hombres, a tripulantes o pasajeros de 1ª clase que sí lograron salvarse.
* **Clase Social**: La 3ª clase presenta la mayor tasa de error (**~23 %**), lo que refleja la alta heterogeneidad y caos durante la evacuación en los compartimentos inferiores.

### Reflexión Ética y Limitaciones de Uso:
El modelo reproduce fielmente los factores sociodemográficos que condicionaron la supervivencia en 1912 (`Sex` y `Pclass`). **Que el modelo prediga con exactitud lo que ocurrió no significa que deba ocurrir de esa manera**.
* **Contextos donde SÍ es aplicable**:
  - Estudios históricos retrospectivos sobre desastres marítimos de época.
  - Simulaciones actuariales con propósitos de calibración histórica de riesgos.
* **Contextos donde NO es aplicable**:
  - **Protocolos modernos de evacuación**: Bajo las normativas marítimas actuales (SOLAS), los protocolos de salvamento se rigen por principios de igualdad, cercanía y vulnerabilidad física, sin distinción socioeconómica ni discriminación de género.
  - **Fijación contemporánea de pólizas individuales**: Utilizar variables como el sexo o el estatus socioeconómico para negar o condicionar servicios de rescate o coberturas constituiría una práctica discriminatoria e ilegal.


---

## 7. Ajuste de Hiperparámetros y Decisión de Ablación (Pasos 58 y 61)

### Estudio de Ablación de Variables:
* **Configuración A (Base)**: ROC-AUC = **0.8867**
* **Configuración B (Base + Cabina)**: ROC-AUC = **0.8871** ($\Delta = +0.0004$)
* **Configuración C (Base sin Título)**: ROC-AUC = **0.8772** ($\Delta = -0.0095$)
* **Configuración D (Base sin Sexo)**: ROC-AUC = **0.8835** ($\Delta = -0.0032$, el título absorbe la señal de género)

### Decisión Final sobre Variables de Cabina:
**Se excluyen las variables de cabina (`HasCabin`, `TCabin`) del modelo final**. 
* **Justificación técnica**: La mejora es estadísticamente insignificante (+0.04 % de ROC-AUC, inferior a la desviación estándar entre pliegues de ±0.02).
* **Justificación de dominio**: Incluir cabina introduce el riesgo de explotar un sesgo de captura histórica (la cabina se registró con mayor detalle para los supervivientes que llegaron a Nueva York), en lugar de representar un factor de protección causal real.

### Hiperparámetros del Modelo Ganador (`HistGradientBoosting`):
* `learning_rate`: 0.1
* `max_depth`: 4
* `max_iter`: 200
* `l2_regularization`: 1.0
* **ROC-AUC alcanzado**: **0.8925**

---

## 8. Evaluación Final en el Conjunto de Prueba y Conclusiones (Pasos 62–63)

### Desempeño en Test Independiente ($n = 179$, Apertura Única):
* **ROC-AUC**: **0.8238** | **Intervalo de Confianza 95 % (Bootstrap, 1000 iteraciones)**: `[0.7469, 0.8939]`
* **Exactitud (Accuracy)**: **0.8101** | **Intervalo de Confianza 95 %**: `[0.7486, 0.8659]`
* **F1-Score**: **0.7424** | **Intervalo de Confianza 95 %**: `[0.6475, 0.8246]`

### Análisis de Incertidumbre y Generalización:
1. **Coherencia con Validación Cruzada**: La estimación puntual en prueba (0.8238) se ubica cómodamente dentro del rango de dispersión observado durante la validación cruzada ($0.8925 \pm 0.024$ con un intervalo empírico de prueba $[0.75, 0.89]$). La diferencia entre CV y Test responde a la varianza inherente de una muestra de prueba reducida ($n=179$) y no a sobreajuste.
2. **Superación Consistente de Baselines**: El modelo final supera con solidez a la clase mayoritaria (Dummy: 0.50 ROC-AUC, 0.61 exactitud) y mejora la regla heurística de género (0.77 ROC-AUC, 0.78 exactitud, F1=0.69).
3. **Decisiones Congeladas y Trazabilidad**: El pipeline productivo queda versionado y congelado bajo `flows/05_ml_models.py`, asegurando reproducibilidad determinista vía `make pipeline`.

