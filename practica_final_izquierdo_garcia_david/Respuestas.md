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
> - duración del incendio: **0.32**
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
En este apartado hemos aplicado una regresión lineal múltiple para tratar de predecir el tamaño de la superficie quemada por un incendio. Como parte del preprocesamiento se han aplicado escalas logarítmicas a variables cuya distribución era muy asimétrica. 
Partiendo del EDA previo se han agrupado algunas variables categóricas, se han eliminado otras y del resultado final se han categorizado para convertir el dataframe en datos numéricos. 
Como consecuencia de la escala logarítmica vemos en el scatterplot de los residuos una línea recta sobre la que se apilan los puntos que corresponde a los incendios con menos de 1 hectárea quemadas. 

---

**Pregunta 2.1** — Indica los valores de MAE, RMSE y R² de la regresión lineal sobre el test set. ¿El modelo funciona bien? ¿Por qué?

> Los valores obtenidos para las métricas del modelo son:
> - $MAE: 0.7691$
> - $RMSE: 1.0119$
> - $R^2: 0.3474$

> Estos valores muestran que el modelo sólo es capaz de predecir ciertos tipos de incendios fallando en los casos más extremos, confirmando que el fenómeno es más complejo que una relación lineal y afectando factores que no estamos teniendo en cuenta en este análisis. 


---

## Ejercicio 3 — Regresión Lineal Múltiple en NumPy

---
Añade aqui tu descripción y analisis:

---

**Pregunta 3.1** — Explica en tus propias palabras qué hace la fórmula β = (XᵀX)⁻¹ Xᵀy y por qué es necesario añadir una columna de unos a la matriz X.

> Como inicialmente tenemos la ecuación matricial $y = X \beta$ y queremos despejar la matriz columna de coeficientes $(\beta)$, tenemos que aplicar la matriz inversa de $X$, que solo es posible calcularla para matrices cuadradas(mismo número de filas que de columnas), y la forma de conseguir matrices cuadradas es multiplicarlas por su traspuesta pues el resultado de la multiplicación de matrices es una matriz con el número de filas de la primera y el número de columnas de la segunda, por lo que esto garantiza que $X^TX$ sea una matriz cuadrada y podemos calcular $(X^TX)^{-1}$

> La secuencia de pasos para obtener la fórmula es la siguiente:

> Partimos de la fórmula $$y =X\beta $$
> Multiplicamos por la izquierda la traspuesta de $X$. $$ X^Ty = X^TX\beta $$
> Volvemos a multiplicar por la izquierda por la inversa de la matriz cuadrada $X^TX$. $$ (X^TX)^{-1}X^Ty = (X^TX)^{-1}(X^TX)\beta $$ 
> Como la multiplicación de una matriz por su inversa es la matriz identidad (elemento neutro en la multiplicación de matrices) obtenemos finalmente la expresión inicial. $$ (X^TX)^{-1}X^Ty = \beta $$

> Esto nos permite calcular los coeficientes a partir de los valores que toman las distintas variables $X$ y el valor de la variable objetivo $y$

> La necesidad e añadir una columna de unos a la matriz $X$ es para poder calcular el coeficiente $\beta_0$ que representa el término independiente puesto que en la expresión anterior estaríamos implementando solamente los coeficientes que acompañan a cada una de las variables y estaríamos limitando el modelo a pasar por el origen de coordenadas y no obtendríamos resultados óptimos.

**Pregunta 3.2** — Copia aquí los cuatro coeficientes ajustados por tu función y compáralos con los valores de referencia del enunciado.

| Parametro | Valor real | Valor ajustado |
|-----------|------------|----------------|
| β₀        | 5.0        |       4.86     |
| β₁        | 2.0        |       2.06     |
| β₂        | -1.0       |       -1.11    |
| β₃        | 0.5        |       0.438    |

> Los resultados obtenidos por el ajuste están bastante próximos a los valores reales, las diferencias caen dentro del error esperado.

**Pregunta 3.3** — ¿Qué valores de MAE, RMSE y R² has obtenido? ¿Se aproximan a los de referencia?

> Los valores de las métricas obtenidas y valores de referencia son los siguientes:

| Métrica | Valor obtenido |   Valor referencia |
|---------|----------------|--------------------|
| $MAE$   |   1.1665       |  $1.2 \pm 0.2$     |
| $RMSE$  |   1.4612       |  $1.5 \pm 0.2$     |
| $R^{2}$ |   0.6897       |  $0.8 \pm 0.05$    |

> El MAE y el RMSE caen dentro del valor esperado, sin embargo el valor $R^2$ queda ligeramente por abajo por lo que el modelo no es tan bueno como se esperaba inicialmente. 

---

## Ejercicio 4 — Series Temporales
---
Anlizamos una serie temporal sintética para extraer la información de la tendencia, estacionalidad y tipo de ruido

---

**Pregunta 4.1** — ¿La serie presenta tendencia? Descríbela brevemente (tipo, dirección, magnitud aproximada).

> Se observa una tendencia creciente que comienza aproximadamente en $50$ y termina en $150$ a lo largo de los seis años de análisis. La pendiente es aproximadamente $m = \frac{150-50}{2024-2018} = 16.7$. Dentro de la tendencia creciente se observa un ciclo mayor de unos tres años.

**Pregunta 4.2** — ¿Hay estacionalidad? Indica el periodo aproximado en días y la amplitud del patrón estacional.

> En el gráfico que muestra la estacionalidad observamos una forma periódica que se repite cada 365 días con valores máximos entorno a $10$ y mínimos alrededor de $-20$, por lo que la estacionalidad tiene una **amplitud de $\textbf{30}$**

**Pregunta 4.3** — ¿Se aprecian ciclos de largo plazo en la serie? ¿Cómo los diferencias de la tendencia?

> Existe un ciclo mayor que no se observa en la gráfica de estacionalidad, pero si observamos con detalle la gráfica de tendencia vemos que no es una línea recta perfecta, sino que tiene cierta oscilación de 3 años entorno a la línea recta.

**Pregunta 4.4** — ¿El residuo se ajusta a un ruido ideal? Indica la media, la desviación típica y el resultado del test de normalidad (p-value) para justificar tu respuesta.

> Del test de Jarque_Bera obtenemos los siguientes resultados: $stat = 1.10$ y $p = 0.57$, por lo que no descartamos la normalidad y observando el histograma de los residuos con la gráfica de la distribución normal superpuesta vemos que se ajusta perfectamente a un ruido ideal. El ruido resenta una distribución nomral de media $\mu = 0.13$ y desviación estándar $\sigma = 3.22$

---

*Fin del documento de respuestas*
