# Respuestas — Práctica Final: Análisis y Modelado de Datos

> Rellena cada pregunta con tu respuesta. Cuando se pida un valor numérico, incluye también una breve explicación de lo que significa.

---

## Ejercicio 1 — Análisis Estadístico Descriptivo
---
En este ejercicio se realiza un EDA sobre un dataset que contiene información sobre incendios forestales en españa desde el año 1968 al año 2016. El objetivo es estudiar la correlación entre las diferentes variables con la superficie quemada en un incendio. Despues de limpiar los datos y la realización de visualizaciones que muestran la distribución y correlación de los datos podemos concluir que los factores que determinan el tamaño de un incendio no están determinados en su totalidad por las condiciones climáticas y ambientales en el momento de producirse. 

---

**Pregunta 1.1** — ¿De qué fuente proviene el dataset y cuál es la variable objetivo (target)? ¿Por qué tiene sentido hacer regresión sobre ella?

> El dataset provine de Kaggle, se puede encontrar en el siguiente [enlace](https://www.kaggle.com/datasets/patrilc/wildfirespain). Proviene de los datos del Ministerio para la transición ecológica y el reto demográfico y está elaborado a partir de los partes de incendios desde los años 1968 al 2016.

> La **variable objetivo** es la **superficie quemada** en cada uno de los incendios, tiene sentido hacer regresión sobre ella ya que es el impacto último de un incendio que va determinado por las condiciones que lo provocan.

**Pregunta 1.2** — ¿Qué distribución tienen las principales variables numéricas y has encontrado outliers? Indica en qué variables y qué has decidido hacer con ellos.

> Las principales variables numéricas presentan distribuciones diversas, por un lado tenemos la temperatura y la humedad relativa en la que podemos observar una distribución normal. Por otro lado existen variables con una distribución muy escorada a la izquierda, es decir, la gran mayoría de los valores son pequeños pero existen valores muy grandes que se separan del grueso de los datos, este es el caso de la duración del incendio, los días desde la última lluvia, la velocidad del viento y la superficie quemada. 

> En el tratamiento previo de los datos para recortar el volumen del dataset he filtrado los datos para los cuales estas variables tienen sentido físico, por ejemplo temperaturas menores de 48ºC (máximo histórico de temperatura máxima en españa) o humedad relativa entre 0 y 100 (cualquier valor fuera de este rango no tiene sentido por la propia definición)

**Pregunta 1.3** — ¿Qué tres variables numéricas tienen mayor correlación (en valor absoluto) con la variable objetivo? Indica los coeficientes.

> En general no se observa una correlación muy fuerte entre la superficie quemada y las variables numéricas (que corresponden a datos meteorológicos), esto indica que en el impacto de un incendio pueden tener un papel más importante otros factores a los estudiados en este apartado. 
> Las variables con mayor correlación con la variable objetivo (superficie quemada) son:
> - duración del incendio:**0.32**
> - días que han pasado desde la última lluvia: **0.10**
> - temperatura máxima del día en que se inicia el incendio: **0.08**

**Pregunta 1.4** — ¿Hay valores nulos en el dataset? ¿Qué porcentaje representan y cómo los has tratado?

> Inicialmente el dataset tiene 585399 filas y 51 columnas ocupando un espacio en memoria de 305.36 MB. Columnas interesantes para este análisis son 14 y de estas columnas hay algunas filas con valores nulos. Eliminamos todas las filas que presentan valores nulos, pues debemos reducir el tamaño del dataset para la práctica y después de este paso nos quedamos con 159161 filas y 14 columnas con un espacio en memoria de 29.70 MB.
> El segundo paso es filtrar estas columnas, pues hay datos que sabemos con certeza que no son verdaderos:
> - Temperaturas superiores a 48ºC
> - Velocidades de viento superiores a 200 Km/h
> - Valores de humedad relativa menores que 0 y mayores que 100
> - Valores de superficie quemada en hectáreas superiores a 600
> - Duración de incendios superiores a 37 días
> - Días desde la última lluvia superiores a 150 días

> Todos estos valores los hemos acotado cotejando con el histórico de valores extremos en España. Cualquier valor que se salga de estos rangos (donde hemos dado bastante margen) es con toda seguridad erróneo debido a errores de lectura de los dispositivos o transcripción
> Despues de eliminar los registros que se encuentran fuera de valores plausibles nuestro dataset contiene **156235 filas y 14 columnas** ocupando un espacio en memoria de 6.26 MB (el dataset exportado en el archivo 'incendios_limpio.parquet' ocupa 1 MB)
---

## Ejercicio 2 — Inferencia con Scikit-Learn

---
Añade aqui tu descripción y analisis:

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> Los valores obtenidos para las métricas del modelo son:
> - MAE: 0.7691
> - RMSE: 1.0119
> - $R^2$: 0.3474

> Estos valores muestran que el modelo sólo es capaz de predecir ciertos tipos de incendios fallando en los casos más extremos, confirmando que el fenómeno es más complejo que una relación lineal y afectando factores que no estamos teniendo en cuenta en este análisis. 


---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
Añade aqui tu descripción y analisis:

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> _Escribe aquí tu respuesta_

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|-----------|----------------|
| β₀        | 5.0       |                |
| β₁        | 2.0       |                |
| β₂        | -1.0      |                |
| β₃        | 0.5       |                |

> _Escribe aquí tu respuesta_

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> _Escribe aquí tu respuesta_

**Pregunta 3.4* — Compara los resultados con la reacción logística anterior para tu dataset y comprueba si el resultado es parecido. Explica qué ha sucedido. 

> _Escribe aquí tu respuesta_

---

## Ejercicio 4 — Series Temporales
---
Añade aqui tu descripción y analisis:

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> _Escribe aquí tu respuesta_

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> _Escribe aquí tu respuesta_

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> _Escribe aquí tu respuesta_

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> _Escribe aquí tu respuesta_

---

*Fin del documento de respuestas*
