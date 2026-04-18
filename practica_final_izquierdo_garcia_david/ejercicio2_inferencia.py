"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 2
Inferencia con Scikit-learn
=============================================================================

DESCRIPCIÓN
-----------
1. Preprocesamiento
    • Aplica las transformaciones necesarias: codificación de variables categóricas
    (LabelEncoder, OneHotEncoder o get_dummies), escalado si procede (StandardScaler o
    MinMaxScaler) y eliminación de columnas que no aporten información.
    • Divide los datos en Train (80 %) y Test (20 %) usando train_test_split(...,
    random_state=42).
    • El preprocesamiento debe estar documentado y justificado en Respuestas.md.

2. Modelo Regresión Lineal (LinearRegression)
    • Entrena el modelo con los datos de entrenamiento.
    • Evalúa sobre el test set calculando: MAE, RMSE y R².
    • Genera el gráfico de residuos (valores predichos en X, residuos en Y).
    • Comenta los resultados en Respuestas.md: ¿el modelo es bueno?, ¿hay overfitting o
    underfitting?, ¿qué variables son más influyentes?

3. Conclusiones
    • Dedica un apartado de Respuestas.md a extraer conclusiones sobre qué información del
    Ejercicio 1 resultó más útil para interpretar los resultados.
=============================================================================
"""

# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

"""
=============================================================================
Aquí se definen las funciones necesarias para el preprocesamiento de los datos, entrenamiento del modelo y evaluación de resultados.
==============================================================================
"""


def transformacion_logaritmica(df, columnas):
    '''
    Aplica una transformación logarítmica a las columnas especificadas y las añade al dataframe.
    Parámetros:
        - df: DataFrame de entrada
        - columnas: lista de nombres de las columnas a transformar
    Output:
        - df: DataFrame con las columnas transformadas
    '''
    df = df.copy()
    for columna in columnas:
        df[f'{columna}_log'] = np.log1p(df[columna])
        df.drop(columns=columna, inplace=True)
    return df

def agrupamiento_variables(df):
    '''
    Agrupa la variable 'tipo_fuego' en superficie y otros.
    Agrupa las categorias de 'tipo_combustible' menores a 1% en 'otros'.
    Parámetros:
        - df: DataFrame de entrada
    Output:
        - df: DataFrame con las variables agrupadas
    '''
    df = df.copy()

    # Agrupamiento de 'tipo_fuego'
    df['tipo_fuego'] = df['tipo_fuego'].apply(lambda x: 'superficie' if x == 'superficie' else 'otros')

    # Agrupamiento de 'tipo_combustible'
    freq_combustible = df['tipo_combustible'].value_counts(normalize=True)
    df['tipo_combustible'] = df['tipo_combustible'].apply(lambda x: x if freq_combustible[x] > 0.01 else 'otros')

    return df

def eliminar_columnas(df, columnas):
    '''
    Elimina las columnas especificadas del dataframe.
    Parámetros:
        - df: DataFrame de entrada
        - columnas: lista de nombres de columnas a eliminar
    Output:
        - df: DataFrame sin las columnas eliminadas
    '''
    df = df.copy()
    df.drop(columns=columnas, inplace=True)
    return df
    
def codificacion_categoricas(df):
    '''
    Aplica codificación One-Hot a las variables categóricas del dataframe.
    Parámetros:
        - df: DataFrame de entrada
    Output:
        - df: DataFrame con las variables categóricas codificadas
    '''
    df = df.copy()
    columnas_categoricas = df.select_dtypes(include=['object', 'category', 'string']).columns
    df = pd.get_dummies(df, columns=columnas_categoricas, drop_first=True, dtype=int)
    return df

def graficar_residuos(y_test, y_pred):
    '''
    Genera un gráfico de residuos.
    Parámetros:
        - y_test: valores reales de la variable objetivo
        - y_pred: valores predichos por el modelo
    Output:
        - Gráfico de residuos guardado en 'output/ej2_residuos.png'
    '''
    residuos = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuos, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Valores Predichos')
    plt.ylabel('Residuos')
    plt.title('Gráfico de Residuos')
    plt.savefig('practica_final_izquierdo_garcia_david/output/ej2_residuos.png')
    plt.close()
"""
=============================================================================
Aquí comienza el main del ejercicio, donde se cargarán los datos limpios, se realizará el preprocesamiento necesario para el modelo
==============================================================================
"""

if __name__ == "__main__":
    # Carga de datos
    df = pd.read_parquet('practica_final_izquierdo_garcia_david/data/incendios_limpio.parquet')

    # Preprocesamiento
    columnas_a_transformar = ['perdidas_superficie_ha', 'dias_ultima_lluvia', 'duracion_dias', 'velocidad_viento_kmh']
    df_modelo = transformacion_logaritmica(df, columnas_a_transformar)
    df_modelo = agrupamiento_variables(df_modelo)
    columnas_a_eliminar = ['clase_dia', 'tipo_ataque']
    df_modelo = eliminar_columnas(df_modelo, columnas_a_eliminar)
    df_modelo = codificacion_categoricas(df_modelo)

    # División en Train y Test
    target = 'perdidas_superficie_ha_log'
    X = df_modelo.drop(columns=target)
    y = df_modelo[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Escalado de variables numéricas
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenamiento del modelo
    modelo = LinearRegression()
    modelo.fit(X_train_scaled, y_train)

    # Predicciones y evaluación
    y_pred = modelo.predict(X_test_scaled)

    # Cálculo de métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Escritura del archivo de resultados
    with open('practica_final_izquierdo_garcia_david/output/ej2_metricas_regresion.txt', 'w') as f:
        f.write(f'MAE: {mae:.4f}\n')
        f.write(f'RMSE: {rmse:.4f}\n')
        f.write(f'R2: {r2:.4f}\n')

    # Coeficientes del modelo para identificar variables influyentes
    # coeficientes = pd.Series(modelo.coef_, index=X_train.columns)
    # coeficientes_ordenados = coeficientes.abs().sort_values(ascending=False)
    # print("Variables más influyentes según los coeficientes del modelo:")
    # print(coeficientes_ordenados.head(10))

    # Gráfico de residuos
    graficar_residuos(y_test, y_pred)
