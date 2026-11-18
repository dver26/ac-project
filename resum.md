# The Dynamics of Fame — Proyecto de Machine Learning

Volem predir la fama d'un tema en un moment. La proposta del Toni és que agafem les dades de quantes persones accedeixen a l'article de Wikipedia d'un cert tema. Tindrem una sèrie temporal per a cada article. La variable que podem observar és el nombre de visites diàries a la pàgina.

El que podríem fer és, sense fer clustering, aplicar un Random Forest o Gradient Boosting a totes les dades per predir el dia següent. Després podem entrenar un model dedicat a cada clúster i comparar quin enfocament ens ha servit més.

El que no sabem és si, quan apliquem clustering, els grups que ens sortiran correspondran a «tipus de fama». Aquesta és una de les preguntes del projecte. Volem veure si apareixen patrons com interès estable, fama fugaç, creixement progressiu o pics recurrents.

## 1. Selección de artículos y descarga de visitas

Trabajaremos inicialmente con **150 artículos de Wikipedia en inglés**, repartidos en cinco ámbitos con 30 artículos por ámbito.

- Deporte
- Cine y televisión
- Música
- Política
- Ciencia y tecnología

Seleccionaremos los artículos antes de examinar sus curvas de visitas. Dentro de cada ámbito, haremos un sorteo con semilla fija entre candidatos que ya existían al comenzar 2023. Para el piloto usaremos **10 artículos, dos por ámbito**.

### Dataset de artículos — `articles.csv`

**Una fila por artículo.** Tendremos 150 filas y cuatro columnas en la primera versión.

| Variable | Qué contiene | Ejemplo |
| --- | --- | --- |
| `page_id` | Identificador para unir las tablas | ID del artículo |
| `title` | Título del artículo en Wikipedia | `Lionel Messi` |
| `domain` | Ámbito del artículo | `deporte` |
| `topic_type` | Qué clase de tema describe | `persona`, `organización`, `evento`, `obra`, `concepto` |

Podemos añadir `created_at` si queremos calcular la antigüedad. `domain` y `topic_type` aportan información distinta. Lionel Messi y la Copa Mundial de Fútbol pertenecen al ámbito `deporte`, pero uno es `persona` y el otro es `evento`.

### Dataset de visitas — `pageviews.csv`

Descargaremos las visitas diarias mediante la **API de Wikimedia Pageviews** con los filtros `all-access`, `user` y `daily`. El periodo será del **1 de enero de 2023 al 31 de diciembre de 2025**, ambos incluidos.

**Una fila por artículo y día.** Con el calendario completo tendremos **150 × 1.096 = 164.400 filas**, contando las filas con visitas ausentes.

| Variable | Qué contiene |
| --- | --- |
| `page_id` | Artículo al que pertenecen las visitas |
| `date` | Fecha de la observación |
| `views` | Número de visitas ese día |

Para cada artículo \(i\), obtenemos la serie \(X_i=(x_{i,1},x_{i,2},\ldots,x_{i,T})\), donde \(x_{i,t}\) son las visitas del artículo \(i\) en el día \(t\).

Antes de calcular las características, alinearemos las series con el mismo calendario y revisaremos los huecos. Conservaremos la diferencia entre los datos ausentes y los ceros cuando podamos establecerla. Guardaremos `views` como conteo original y calcularemos \(z_{i,t}=\log(1+x_{i,t})\) para las características que usan escala logarítmica.

## 2. Variables para K-Means

El objetivo es describir cómo se ha comportado históricamente cada artículo. Para esta fase usaremos **solo las visitas de 2023**.

Per cada article podem fer una regressió lineal i treure el seu pendent. Això és la **tendència**. Però si hi ha un pic com una muntanya i després les visites tornen al nivell habitual, el pendent pot quedar a prop de zero. Per això afegim variables que mesuren la intensitat, la freqüència i la durada dels pics.

### A. Tendencia global — `trend`

**¿El artículo está ganando o perdiendo popularidad a largo plazo?**

Ajustamos una recta sobre la serie logarítmica \(z_t\) y guardamos su pendiente. Usaremos **Theil–Sen** para reducir la influencia de las observaciones extremas. Una pendiente positiva indica crecimiento, una negativa indica descenso y una cercana a cero indica poca tendencia global.

### B. Volatilidad — `volatility_iqr`

**¿Cuánto fluctúan habitualmente las visitas?**

Calculamos el rango intercuartílico de la serie logarítmica. Es decir, \(Q_{0.75}(z)-Q_{0.25}(z)\). Esto mide la dispersión de la parte central de la serie y reduce la influencia de los valores extremos.

### C. Intensidad del pico máximo — `peak_intensity`

**¿Cuánto ha destacado el momento de máxima atención respecto al nivel habitual?**

Calculamos una media móvil de 7 días y dividimos su máximo entre la mediana de las visitas diarias de 2023. Si dos artículos suelen tener 100 visitas al día y sus máximos suavizados son 150 y 2.000, obtendríamos intensidades de 1,5 y 20.

En el piloto revisaremos los artículos con mediana cero antes de fijar el tratamiento de esta variable.

### D. Número de picos importantes — `peak_count_per_year`

**¿Cuántas explosiones de atención ha experimentado el artículo?**

Suavizamos la serie y detectamos máximos locales con `scipy.signal.find_peaks`. Exigiremos una prominencia mínima y cierta separación entre picos. La prominencia mide cuánto sobresale un pico respecto a su entorno.

Guardaremos el número de picos por año observado. Como todos los artículos usan el mismo año para K-Means, equivale al número de picos detectados durante 2023.

### E. Persistencia a siete días — `autocorr_7`

**¿Se parecen las visitas a las de siete días antes?**

Quitamos la tendencia global a la serie logarítmica y calculamos la correlación entre los residuos de cada día y los de siete días antes. Una correlación alta indica persistencia a ese retardo. Al interpretar los grupos, revisaremos las curvas para ver si esa persistencia coincide con un patrón semanal de calendario.

### F. Duración de los episodios de popularidad — `mean_episode_days`

**¿Cuántos días permanece el artículo por encima de su nivel habitual cuando recibe más atención?**

Comparamos la serie suavizada con un umbral relativo a la mediana histórica del artículo. Identificamos los tramos consecutivos por encima del umbral y guardamos su duración media en días.

Partimos de 7 días de suavizado. Durante el piloto podemos probar también 3 días para ver si así conservamos mejor los picos breves. La prominencia, la separación entre picos y el umbral de los episodios se concretarán en el piloto.

## 3. Forma del dataset de clustering

**Una fila por artículo.** El archivo `cluster_features.csv` tendrá **150 filas y siete columnas**. Una columna identifica el artículo y las otras seis son las variables de K-Means.

| `page_id` | `trend` | `volatility_iqr` | `peak_intensity` | `peak_count_per_year` | `autocorr_7` | `mean_episode_days` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ID de Messi | 0.01 | 0.2 | 2.1 | 3 | 0.7 | 8 |
| ID de Película A | -0.02 | 1.1 | 12.5 | 1 | 0.2 | 15 |
| ID de Bitcoin | 0.01 | 0.8 | 5.4 | 6 | 0.5 | 5 |

*Valores ficticios para ilustrar la estructura.*

Estandarizaremos las seis variables para que sus escalas sean comparables. La **matriz de entrada a K-Means es de 150 × 6**. Probaremos varios valores de K y miraremos el coeficiente de silueta, la estabilidad de los grupos y las curvas de artículos representativos.

La salida será `clusters.csv`, con **150 filas y dos columnas**:

| Variable | Qué contiene |
| --- | --- |
| `page_id` | Artículo |
| `cluster` | Etiqueta del clúster asignado con los datos de 2023 |

Los grupos se mantendrán fijos para la comparación predictiva. El ámbito y el tipo de tema se usarán como variables del predictor y también pueden servir para describir los grupos una vez formados.

## 4. Dataset para predecir las visitas del día siguiente

El predictor trabaja con **una fila por artículo y fecha de predicción**. Para un artículo \(i\), al cierre del día \(t\) queremos predecir \(x_{i,t+1}\). Las variables de visitas se calculan por artículo y con datos disponibles hasta \(t\), incluido ese día.

### Variables del predictor

| Variables | Qué representan |
| --- | --- |
| `lag_1` a `lag_7` | Las siete visitas diarias anteriores a la fecha objetivo. Si el objetivo es \(t+1\), `lag_1` son las visitas de \(t\) y `lag_7` las de \(t-6\). |
| `mean_3`, `mean_7`, `mean_30` | Medias de visitas de los últimos 3, 7 y 30 días, incluido \(t\). |
| `std_7` | Desviación típica de las visitas de los últimos 7 días. |
| `max_7` | Máximo de visitas de los últimos 7 días. |
| `target_day_of_week` | Día de la semana de la fecha que queremos predecir. |
| `domain`, `topic_type` | Ámbito y tipo de tema, unidos desde `articles.csv` por `page_id`. |
| `article_age_days` | Opcional. Días entre la fecha de creación y el día \(t\). |

El ámbito y el tipo se repiten en todas las filas de un mismo artículo. Las visitas recientes cambian cada día. El tamaño del texto se podría incorporar en una segunda versión si podemos obtener su longitud en la fecha de cada predicción.

### Forma de `prediction_dataset.csv`

| Bloque | Columnas |
| --- | --- |
| Identificación y fechas | `page_id`, `date`, `target_date` |
| Visitas recientes | `lag_1` a `lag_7`, `mean_3`, `mean_7`, `mean_30`, `std_7`, `max_7` |
| Calendario y artículo | `target_day_of_week`, `domain`, `topic_type` |
| Objetivo | `target_views`, las visitas reales de `target_date` |

La primera versión tiene **19 columnas, de las cuales 15 son variables del predictor antes de codificar las categóricas**. Para las fechas objetivo de 2024 y 2025 tendremos hasta **109.650 filas** si todas las observaciones necesarias están disponibles.

Por ejemplo, la fila de Messi con `date = 2025-06-30` tiene `target_date = 2025-07-01`. Sus medias y retardos usan visitas hasta el 30 de junio. `target_views` contiene las visitas reales del 1 de julio y se usa como etiqueta para entrenar o evaluar.

`page_id` permite unir las tablas. Al unir `clusters.csv`, el modelo especializado usa `cluster` para seleccionar qué Random Forest debe predecir cada fila. Ambos enfoques reciben las mismas variables predictivas.

## 5. Partición temporal

| Periodo | Qué hacemos | Unidad y tamaño máximo |
| --- | --- | --- |
| **2023** | Calcular las seis características, ajustar el escalado y K-Means y fijar los grupos. | 150 artículos |
| **2024** | Entrenar los predictores con fechas objetivo de ese año. | 54.900 filas artículo-día |
| **Enero–junio de 2025** | Validar y ajustar los predictores. | 27.150 filas artículo-día |
| **Julio–diciembre de 2025** | Comparar los modelos en el test final. | 27.600 filas artículo-día |

La partición se hace por **fecha objetivo**. Las ventanas pueden usar días de un periodo anterior. Por ejemplo, para predecir el 1 de enero de 2025 podemos usar visitas de diciembre de 2024. Los datos del test se reservan para calcular el rendimiento final.

## 6. Modelos y dataset de resultados

Compararemos primero estas dos configuraciones, con las mismas variables, fechas y métricas.

1. **Modelo global**. Un Random Forest entrenado con todos los artículos.
2. **Modelos especializados**. Un Random Forest por clúster. Cada uno se entrena con las filas de los artículos de su grupo.

Como referencias sencillas, compararemos con usar las visitas del último día disponible y las del mismo día de la semana anterior. Más adelante podemos probar Gradient Boosting o un ensemble de los modelos especializados.

Para evaluar, guardaremos **una fila por artículo y fecha objetivo**, con las predicciones alineadas para comparar exactamente los mismos casos.

| Variable | Qué contiene |
| --- | --- |
| `page_id`, `target_date` | Artículo y fecha evaluada |
| `cluster` | Clúster del artículo |
| `views_actual` | Visitas observadas |
| `pred_global` | Predicción del Random Forest global |
| `pred_cluster` | Predicción del Random Forest del clúster |
| `pred_last_day` | Predicción de referencia con las visitas del día anterior |
| `pred_last_week` | Predicción de referencia con las visitas del mismo día de la semana anterior |

Calcularemos **MAE y RMSE** sobre las mismas filas de test. También podemos desglosar el error por clúster para ver si la especialización ayuda en unos patrones más que en otros.

## 7. Próximo paso

Hacer el piloto con **10 artículos, dos por ámbito**. Descargar las series de los tres años, revisar fechas y huecos, dibujar las curvas e implementar en Python la función que devuelve las seis características de clustering.

Después, comprobar con ejemplos que cada fila diaria tiene la fecha objetivo correcta y que las medias y retardos usan exclusivamente visitas previas a esa fecha. Con el flujo validado, repetir el proceso con 150 artículos y realizar la comparación entre el modelo global y los modelos por clúster.
