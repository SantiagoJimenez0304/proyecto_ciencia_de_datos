# Informe de Análisis Exploratorio de Datos (EDA)
**Proyecto:** Análisis de Supervivencia - RMS Titanic  
**Destinatario:** Gerencia de Riesgos y Actuarial  
**Estado:** Finalizado  

---

## 1. Resumen Ejecutivo y Hallazgos Principales

El análisis exploratorio sobre los 891 pasajeros con registro completo de supervivencia revela patrones estructurales determinantes en las probabilidades de evacuación y supervivencia:

1. **La regla de protocolo ("Mujeres y niños primero") fue el factor dominante:**
   - La tasa de supervivencia en mujeres fue del **74.20 %** ($n = 314$), frente al **18.89 %** en hombres adultos ($n = 577$).
   - La prueba chi-cuadrado ($\chi^2 = 260.72$, $p < 10^{-50}$, $gl = 1$) confirma una dependencia estadística abrumadora que supera cualquier variación atribuible al azar.
   - En niños menores de 16 años ($n = 83$), la tasa de supervivencia alcanzó el **59.04 %**, con un marcado beneficio para los varones infantes en comparación con los hombres adultos.

2. **Gradiente socioeconómico pronunciado por clase de pasaje (`Pclass`):**
   - Primera clase: **62.96 %** de supervivencia ($n = 216$).
   - Segunda clase: **47.28 %** de supervivencia ($n = 184$).
   - Tercera clase: **24.24 %** de supervivencia ($n = 491$).
   - Un pasajero de 1ª clase tuvo 2.6 veces más probabilidad de sobrevivir que uno de 3ª clase.

3. **Interacción crítica entre género y clase social:**
   - Las mujeres de 1ª y 2ª clase tuvieron tasas de supervivencia casi totales: **96.81 %** ($n = 94$) y **92.11 %** ($n = 76$), respectivamente.
   - En 3ª clase, la supervivencia femenina cayó abruptamente al **50.00 %** ($n = 144$), evidenciando que las barreras físicas y la ubicación de los camarotes en cubiertas inferiores neutralizaron parcialmente el privilegio del protocolo.
   - En los hombres, pertenecer a 1ª clase elevó la supervivencia al **36.89 %** ($n = 122$), frente al **15.74 %** en 2ª clase ($n = 108$) y **13.54 %** en 3ª clase ($n = 347$).

4. **Efecto protector moderado del tamaño familiar (`Familia`):**
   - Pasajeros que viajaban solos (`Alone`, $n = 537$): **30.35 %** de supervivencia.
   - Familias pequeñas de 2 a 4 miembros ($n = 314$): **57.88 %** de supervivencia promedio (pico en familias de 4 miembros con **72.41 %**, $n = 29$).
   - Familias numerosas de 5 o más integrantes ($n = 40$): caída drástica al **15.00 %**, debido a la dificultad extrema de reunir a todo el grupo en medio del caos antes de abordar los botes salvavidas.

---

## 2. Decisiones de Limpieza, Transformación y sus Sesgos Asociados

Toda manipulación de datos introduce supuestos que alteran la distribución muestral. Documentamos las siguientes cuatro decisiones:

1. **Imputación de Edad (`Age`):**
   - Se registraron 177 valores faltantes (19.87 % del total).
   - En la capa Silver se imputó con la media por género (mujeres: 27.91 años, hombres: 30.72 años).
   - **Impacto y sesgo:** La desviación estándar global se redujo de **14.53 años** (datos observados) a **13.01 años** (post-imputación). Se generó una concentración artificial de densidad en torno a las medias de grupo. Por este motivo, el modelo de machine learning no utiliza la columna pre-imputada en Silver, sino que entrena un transformador desacoplado (`GroupAgeImputer`) exclusivamente sobre `X_train` con estratos por género y clase social.
2. **Consolidación del título `Master` a `Mr`:**
   - Históricamente, `Master` designaba a niños varones pertenecientes a familias de clase media/alta.
   - Al fusionar `Master` en `Mr` para evitar cardinalidad dispersa en modelado lineal, se corría el riesgo de enmascarar a los niños varones.
   - **Mitigación:** La variable de ingeniería `Person` (`child` si `Age < 16`) rescata explícitamente a este subgrupo, garantizando que el modelo reconozca la protección especial brindada a la infancia.
3. **Tratamiento de Cabina (`Cabin`):**
   - El 77.10 % de los pasajeros carece de número de cabina registrado ($n = 687$).
   - Se creó la categoría `'X'` para nulos y el indicador binario `HasCabin`.
   - **Riesgo:** El registro de cabina está fuertemente correlacionado con la supervivencia ($66.67 \%$ en quienes tienen cabina vs. $29.99 \%$ en los que no), no solo por ubicación espacial en el barco, sino porque a los supervivientes se les pudo entrevistar y documentar con mayor exhaustividad tras el rescate (*sesgo de registro por supervivencia*). Debido a esto, se resolvió no utilizar variables de cabina en el modelo productivo.
4. **Puerto de Embarque (`Embarked`):**
   - Solo 2 pasajeros carecían de puerto; se imputaron con la moda (`'S'` - Southampton), lo cual no distorsiona las estimaciones globales.

---

## 3. Confusiones Detectadas y su Aislamiento Analítico

### Paradoja de Puerto de Embarque vs. Clase Social
Al evaluar la tasa bruta de supervivencia por puerto de embarque:
- **Cherburgo (`C`):** 55.36 % ($n = 168$)
- **Queenstown (`Q`):** 38.96 % ($n = 77$)
- **Southampton (`S`):** 33.70 % ($n = 644$)

Una lectura ingenua sugeriría que abordar en Cherburgo otorgaba una ventaja directa de supervivencia. Sin embargo, al estratificar por `Pclass`:
- En 1ª clase: Cherburgo 65.22 %, Southampton 58.27 %.
- En 2ª clase: Cherburgo 52.94 %, Southampton 46.34 %.
- En 3ª clase: Cherburgo 37.88 %, Southampton 18.98 %.

El **50.60 %** de los pasajeros que abordaron en Cherburgo viajaban en 1ª clase, comparado con apenas el **21.35 %** en Southampton y el **2.60 %** en Queenstown. La aparente asociación puerto-supervivencia era en realidad una **confusión por composición socioeconómica**.

---

## 4. Qué NO Podemos Afirmar (Límites Metodológicos y Advertencias)

1. **Asociación no es Causalidad:**  
   Pagar un pasaje más caro (`Fare`) no incrementó físicamente la flotabilidad ni blindó al pasajero contra el agua gélida; facilitó acceso a camarotes ubicados en cubiertas superiores (`Boat Deck`), proximidad física inmediata a los pescantes y una menor cantidad de rejas y mamparas que sortear durante el hundimiento.
2. **Suceso Histórico Único y No Generalizable:**  
   Los datos pertenecen a un único naufragio ocurrido la noche del 14 al 15 de abril de 1912 bajo circunstancias irrepetibles: mar en calma absoluta, temperaturas bajo cero, escasez reglamentaria de botes salvavidas (solo para el 52 % del aforo total) y una estricta adhesión a códigos de caballería de la época eduardiana.
3. **Inaplicabilidad a Emergencias Contemporáneas:**  
   Este análisis y los modelos derivados no deben utilizarse bajo ningún concepto para tarificar pólizas actuales de vida ni para guiar protocolos modernos de evacuación marítima o aeronáutica, los cuales se rigen por principios universales de evacuación por zonas y disponibilidad de plazas para el 100 % de los ocupantes.
