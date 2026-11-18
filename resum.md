# The Dynamics of Fame — Proyecto de Machine Learning

Estoy haciendo un proyecto universitario con machine learning clásico para estudiar cómo cambia la popularidad de un tema. Usamos las **visitas diarias a su artículo de Wikipedia** como medida de popularidad.

## Objetivos

1. Explorar si aparecen grupos de artículos con patrones de popularidad distintos, como interés estable, picos breves, picos recurrentes o crecimiento progresivo. Estos son ejemplos de patrones posibles. Queremos ver qué grupos salen de los datos.
2. Predecir las visitas del día siguiente y comprobar si entrenar **un modelo por clúster** mejora el rendimiento de **un único modelo global**.

## Dataset previsto

- **150 artículos de Wikipedia en inglés**, repartidos en cinco ámbitos con unos 30 artículos por ámbito. Los ámbitos propuestos son deporte, cine y televisión, música, política, y ciencia y tecnología.
- Seleccionaremos los artículos con una semilla fija antes de examinar sus curvas. Usaremos artículos que ya existían al empezar el periodo.
- Descargaremos las **visitas diarias del 1 de enero de 2023 al 31 de diciembre de 2025** mediante la API de Wikimedia Pageviews, con los filtros `all-access`, `user` y `daily`.
- `articles.csv` tendrá una fila por artículo con `page_id`, `title`, `domain` y `topic_type`. El tipo indica si el artículo trata sobre una persona, organización, evento, obra o concepto, por ejemplo. La fecha de creación es opcional.
- `pageviews.csv` tendrá una fila por artículo y día con `page_id`, `date` y `views`. Conservaremos los huecos de la descarga para revisar su origen antes de tratarlos.

## Clustering

Cada artículo será un punto en K-Means. Calcularemos seis características con sus visitas de **2023**:

1. **Tendencia**. Pendiente de Theil–Sen sobre `log(1 + visitas)`.
2. **Volatilidad**. Rango intercuartílico de las visitas en escala logarítmica.
3. **Intensidad del pico máximo**. Máximo de la media móvil dividido por la mediana histórica del artículo.
4. **Número de picos relevantes**. Máximos locales detectados tras suavizar la serie, con una prominencia y separación mínimas.
5. **Persistencia a siete días**. Autocorrelación con retardo 7 sobre la serie logarítmica, tras eliminar su tendencia.
6. **Duración media de los episodios de atención elevada**. Días consecutivos en los que la serie suavizada supera un umbral relativo al nivel habitual del artículo.

Estandarizaremos las seis variables antes de aplicar K-Means. Probaremos varios valores de K y valoraremos el silhouette, la estabilidad y las curvas de artículos representativos de cada grupo. Los grupos se definirán solo con las características de visitas y quedarán fijos durante la comparación de predictores.

## Predicción de visitas

Cada fila del dataset del predictor representa **un artículo en un día `t`**. El objetivo es predecir las visitas de ese artículo en **`t+1`** con la información disponible hasta `t`.

Variables iniciales:

- Visitas de cada uno de los últimos 7 días, incluido `t`.
- Medias de los últimos 3, 7 y 30 días.
- Desviación típica y máximo de los últimos 7 días.
- Día de la semana de `t+1`.
- Ámbito (`domain`) y tipo de tema (`topic_type`), unidos por `page_id`.
- Opcionalmente, antigüedad del artículo en la fecha de predicción. El tamaño del texto se podría añadir más adelante si tenemos su longitud histórica.

## Diseño experimental

- **2023** — Calcular las características históricas, ajustar el escalado y K-Means y asignar los artículos a sus clústeres.
- **2024** — Entrenar un Random Forest global y un Random Forest para cada clúster.
- **Enero–junio de 2025** — Validar y ajustar los predictores.
- **Julio–diciembre de 2025** — Evaluar la comparación final.

El modelo global y los modelos por clúster usarán las mismas variables, fechas y métricas. En el enfoque por clúster, cada artículo se predice con el modelo de su grupo. Como referencia, compararemos también con predecir las visitas del último día disponible o las del mismo día de la semana anterior. Usaremos MAE y RMSE. Si la comparación principal está terminada, podemos probar un ensemble de los modelos especializados o Gradient Boosting.

La evaluación será cronológica. Para cada predicción, las ventanas de visitas solo pueden contener días ya observados. Por ejemplo, para predecir el 1 de enero de 2025 sí podemos usar los últimos días de diciembre de 2024.

## Primer paso

Hacer un piloto con 10 artículos, dos por ámbito. Descargar sus series, revisar fechas y huecos, dibujar las curvas y comprobar que las seis características se comportan como esperamos. Después, repetir el proceso con los 150 artículos.

Usa este planteamiento como contexto para ayudarme a desarrollar el proyecto en Python paso a paso. Distingue las decisiones ya acordadas de los detalles todavía por fijar, como el umbral de los picos y el número de clústeres.
