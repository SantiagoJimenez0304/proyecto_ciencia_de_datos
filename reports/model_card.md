# Ficha del Modelo (Model Card): Titanic Survivor Predictor v1

## 1. Información General y Propósito

| Campo | Detalle |
| :--- | :--- |
| **Nombre del modelo** | `titanic_model_v1` |
| **Versión** | 1.0.0 |
| **Fecha de emisión** | Noviembre 2026 |
| **Tipo de modelo** | Pipeline scikit-learn con `HistGradientBoostingClassifier` |
| **Licencia / Uso previsto**| Ejercicio académico y benchmark de ciencia de datos sobre supervivencia histórica |
| **Autoría** | Equipo de Ciencia de Datos e Ingeniería |

### Propósito y Uso Previsto
Estimar la probabilidad a posteriori de supervivencia de un pasajero a bordo del RMS Titanic a partir de variables sociodemográficas y de boletería registradas al momento del embarque (abril de 1912).

---

## 2. Datos de Entrenamiento y Auditoría

- **Fuente de datos:** Capa Silver generada desde la fuente oficial del taller (`data/silver/titanic_silver.parquet`).
- **Registros totales:** 891 pasajeros con etiqueta de supervivencia observada.
- **División de datos:** Partición estratificada 80/20 (712 entrenamiento / 179 prueba) para validación de desarrollo; entrenamiento del artefacto productivo final con $n = 891$.
- **Huella digital de datos (`data_sha256`):** `c2e1d0b71949330f842c84bd83e033c102de12ae8d2f9b55a2add62922554187`
- **Huella digital del modelo (`model_sha256`):** `ef74a9298b4f416bdd092f3610f8ee0ea43756aa36af858ce148417db5ce8235`

---

## 3. Variables y Arquitectura del Pipeline

### Variables Utilizadas
1. **Numéricas (5):**
   - `Age` (imputada en entrenamiento mediante `GroupAgeImputer` por estrato `Sex` × `Pclass`).
   - `SibSp` (número de cónyuges y hermanos a bordo).
   - `Parch` (número de padres e hijos a bordo).
   - `Fare` (tarifa del pasaje en libras esterlinas).
   - `Familia` (`SibSp` + `Parch`).
2. **Categóricas (5):**
   - `Pclass` (clase del billete: 1ª, 2ª o 3ª).
   - `Sex` (`female`, `male`).
   - `Embarked` (`S`, `C`, `Q`).
   - `TTitulo` (`Mr`, `Mrs`, `Miss`, `Otro`).
   - `Alone` (`Alone`, `with Family`).

### Variables Excluidas y Justificación
- **Cabina (`HasCabin`, `TCabin`):** Excluidas por *sesgo de registro por supervivencia*. El estudio de ablación evidenció un aporte marginal insignificante (+0.0004 ROC-AUC), por lo que se descartaron para priorizar la parsimonia.
- **Identificadores y Texto Libre (`PassengerId`, `Name`, `Ticket`):** Cardinalidad excesiva y riesgo directo de sobreajuste.
- **Etiquetas de salida (`Survivor`):** Fuga de datos evidente.

### Hiperparámetros Óptimos Congelados
- `model__learning_rate`: `0.1`
- `model__max_depth`: `4`
- `model__max_iter`: `200`
- `model__l2_regularization`: `1.0`
- **Umbral de decisión (`threshold`):** `0.50` (equilibra precisión y exhaustividad con calibración probabilística Brier Score = 0.1302).

---

## 4. Métricas de Desempeño e Incertidumbre

### Validación Cruzada (Entrenamiento, 5-Fold Estratificado, $n = 712$)
- **ROC-AUC:** $0.8925 \pm 0.0241$
- **Exactitud:** $0.8357 \pm 0.0352$
- **F1-Score:** $0.7792 \pm 0.0431$

### Conjunto de Prueba Independiente ($n = 179$, Apertura Única)
- **ROC-AUC:** `0.8238` | **IC 95% Bootstrap (1000 remuestreos):** `[0.7469, 0.8939]`
- **Exactitud:** `0.8101` | **IC 95% Bootstrap:** `[0.7486, 0.8659]`
- **F1-Score:** `0.7424` | **IC 95% Bootstrap:** `[0.6475, 0.8246]`

> **Nota sobre Incertidumbre:** Con un conjunto de prueba de 179 observaciones, la amplitud del intervalo de confianza para ROC-AUC es de casi 0.15 puntos porcentuales. Esto confirma que comparar modelos basándose exclusivamente en diferencias pequeñas de prueba puntual es metodológicamente engañoso.

### Comparativa vs. Baselines en Test
| Modelo | ROC-AUC (Test) | Exactitud (Test) | F1-Score (Test) |
| :--- | :---: | :---: | :---: |
| **Dummy (Clase Mayoritaria)** | 0.5000 | 0.6145 | 0.0000 |
| **Regla de Género (Mujeres=1)** | 0.7703 | 0.7877 | 0.6923 |
| **HistGradientBoosting (Final)** | **0.8238** | **0.8101** | **0.7424** |

---

## 5. Desempeño por Subgrupos Sociodemográficos

El diagnóstico de errores fuera de muestra sobre $X_{\text{train}}$ muestra:
- **Por Género:** Tasa de error en mujeres: 19.37 % ($n = 253$); en hombres: 17.21 % ($n = 459$).
- **Por Clase Social:**
  - 1ª Clase: 21.05 % de error ($n = 171$, principalmente hombres adinerados que sobrevivieron contra el patrón general).
  - 2ª Clase: 8.67 % de error ($n = 150$, grupo más nítidamente separado por la regla de género).
  - 3ª Clase: 20.20 % de error ($n = 391$, principalmente mujeres de familias numerosas que no lograron abordar los botes).

---

## 6. Limitaciones, Riesgos y Usos No Recomendados

1. **Restricción a Contexto Histórico Único:**  
   Este modelo refleja las normas culturales de 1912 y la logística de evacuación de un transatlántico de vapor específico. No generaliza a otros siniestros náuticos ni aeronáuticos.
2. **Prohibición de Uso Discriminatorio o Actuarial Moderno:**  
   Bajo ningún concepto este modelo debe emplearse para tarificación de seguros, asignación de recursos médicos de emergencia ni protocolos de rescate contemporáneos.
3. **Dependencia de Variables Sensibles:**  
   Las variables `Sex` y `Pclass` son los predictores primarios del modelo, lo cual es inaceptable para sistemas de decisión automatizada en la actualidad.

---

## 7. Entorno Técnico y Reproducibilidad

- **Python:** 3.14.4
- **scikit-learn:** 1.9.1
- **pandas:** 3.1.0
- **joblib:** 1.5.3

### Reproducción Completa
Para ejecutar todo el pipeline desde la ingesta hasta el modelo final guardado:
```bash
make pipeline
```
O de manera equivalente:
```bash
python -m flows.job --stage all
```
Para ejecutar la suite completa de pruebas unitarias:
```bash
make test
```
