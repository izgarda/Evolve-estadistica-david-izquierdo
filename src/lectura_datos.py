import pandas as pd
import numpy as np
import os


print(os.getcwd())

datos = pd.read_csv('Evolve-estadistica-david-izquierdo/data/spain_wildfires/archive/incendios.csv', sep=';')

# print(datos.head())

# print(datos.describe())

# print('Nulos por columna')
# print(datos.isnull().sum())

# print('Nulos por columna (porcentaje)')
# print((datos.isnull().sum()/len(datos))*100)

columnas_seleccionadas = [
    'anio', 'probabilidadignicion', 'idpeligro', 'comunidad', 'provincia', 
    'nummunicipiosafectados', 'horadeteccion', 'mesdeteccion', 'duracion', 
    'primeranotificaciondesde112', 'iddetectadopor', 'idcausa', 'idclasedia', 
    'diasultimalluvia', 'tempmaxima', 'humrelativa', 'velocidadviento', 
    'perdidassuperficiales', 'numeromediospersonal', 'numeromediospesados', 
    'numeromediosaereos', 'lugar', 'combustible', 'tipodefuego', 
    'tipodeataque', 'claseincendio', 'intencionalidad'
]
df_recortado = datos[columnas_seleccionadas].copy()
print(f"Dimensiones del nuevo dataset: {df_recortado.shape}")

# Calculo de nulos por año en datos meteorologicos

nulos_por_anio_pct = df_recortado.isnull().groupby(df_recortado['anio']).mean() * 100

columnas_clima = ['tempmaxima', 'humrelativa', 'velocidadviento', 'diasultimalluvia']
evolucion_clima = nulos_por_anio_pct[columnas_clima]

print("Porcentaje de nulos por año en variables climáticas:")
print(evolucion_clima)

# Eliminación de filas con nulos en datos meteorologicos

df_limpio = df_recortado.dropna(subset=columnas_clima).copy()

print(f"Filas originales: {len(df_recortado)}")
print(f"Peso en memoria original: {datos.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
print(f"Filas limpias: {len(df_limpio)}")
print(f"Peso en memoria limpio: {df_limpio.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

print('Nulos por columna después de la limpieza (porcentaje)')
print((df_limpio.isnull().sum()/len(df_limpio))*100)