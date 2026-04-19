"""
=============================================================================
PRÁCTICA FINAL — EJERCICIO 1
Análisis estadístico descriptivo
=============================================================================

DESCRIPCIÓN
-----------
A) Resumen estructural
    • Número de filas, columnas y tamaño en memoria.
    • Tipos de dato de cada columna (dtypes).
    • Porcentaje de valores nulos por columna y decisión de tratamiento.

B) Estadísticos descriptivos de variables numéricas
    • Media, mediana, moda, desviación típica, varianza, mínimo, máximo y cuartiles.
    • Rango intercuartílico (IQR) de la variable objetivo.
    • Coeficiente de asimetría (skewness) y curtosis para al menos la variable objetivo.

C) Distribuciones
    • Histogramas de todas las variables numéricas
    • Boxplots de la variable objetivo, segmentados por cada variable categórica.
    • Detección y tratamiento de outliers (método IQR o Z-score; justifica cuál usas).

D) Variables categóricas
    • Frecuencia absoluta y relativa de cada categoría.
    • Gráfico de barras o de sectores para cada variable categórica.
    • Análisis de si alguna categoría domina el dataset (desbalance).

E) Correlaciones
    • Mapa de calor (heatmap) de la matriz de correlaciones de Pearson de las variables
    numéricas.
    • Identificación de las tres variables con mayor correlación (en valor absoluto) con la variable
    objetivo.
    • Detección de posible multicolinealidad entre predictoras (pares con |r| > 0,9).
=============================================================================
"""

# Importación de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""
=============================================================================
Definición de funciones para el análisis descriptivo, para mantener el código organizado y modularizado.
=============================================================================
"""

"""
=============================================================================
Función para generar histogramas con tendencia KDE de las variables numéricas del DataFrame.
=============================================================================
"""
def histogramas_variables_numericas(df) -> None:
    '''
    Función para generar histogramas con tendencia KDE de las variables numéricas del DataFrame.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene las variables numéricas a graficar
    Output:
        Guarda una imagen PNG con los histogramas en la carpeta output.
    '''
    # Seleccion columnas numéricas
    columnas_numericas = df.select_dtypes(include=['number']).columns

    # cuadríacula para 6 gráficos (2 filas x 3 columnas)
    columnas_grid = 3
    filas_grid = 2

    # Figura base y subplots
    fig, axes = plt.subplots(nrows=filas_grid, ncols=columnas_grid, figsize=(15, 5 * filas_grid))

    # Aplanamos la matriz de ejes para poder iterar sobre ella fácilmente con un bucle for
    axes = axes.flatten()

    # Título general para toda la figura
    fig.suptitle('Distribución de Variables Numéricas con Tendencia KDE', fontsize=14, fontweight='bold', y=1)

    # diccionarios para ajustar bin y kde en función de la variable, para mejorar la visualización de cada histograma
    diccionario_bins = {
        'duracion_dias': 15,
        'temp_maxima_c': 25,
        'humedad_relativa_pct': 25,
        'dias_ultima_lluvia': 60,
        'velocidad_viento_kmh': 60,
        'perdidas_superficie_ha': 80
    }
    diccionario_kde = {'duracion_dias': False}

    # Bucle para dibujar cada histograma en su recuadro correspondiente
    for i, col in enumerate(columnas_numericas):

        sns.histplot(data=df,
                    x=col,
                    kde=diccionario_kde.get(col,True),
                    ax=axes[i], color='steelblue',
                    bins=diccionario_bins.get(col, 40),
                    edgecolor='black',
                    stat='proportion')
        
        axes[i].set_xlabel(col, fontsize=16)
        axes[i].set_ylabel('Frecuencia relativa')

    # Espacio entre subplots
    plt.tight_layout(h_pad=4.0, w_pad=2.0)

    # Guardado de la imagen
    ruta_guardado = 'practica_final_izquierdo_garcia_david/output/ej1_histogramas.png'
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')

"""
=============================================================================
Función para generar boxplots de la variable objetivo 'perdidas_superficie_ha' segmentados por cada variable categórica.
=============================================================================
"""

def boxplot_variable_objetivo_por_categoria(df, columnas, ruta_guardado, x_size = 20) -> None:
    '''
    Función para generar boxplots de la variable objetivo 'perdidas_superficie_ha' segmentados por cada variable categórica.
    Parámetros:
        df (pd.DataFrame): DataFrame que contiene todos los datos
        columnas (list): Lista de nombres de columnas categóricas
        ruta_guardado (str): Ruta donde se guardará la imagen
    Output:
        Guarda una imagen PNG con los boxplots en la carpeta output.
    '''
    target = 'perdidas_superficie_ha'

    # Configurar la cuadrícula 2x2
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(x_size, 15))
    axes = axes.flatten()

    fig.suptitle(f'Boxplots de {target} por Categoría (Escala Logarítmica)', 
                fontsize=20, fontweight='bold')
    
    # orden de categorías temporales
    orden_meses = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    orden_hora = ['manana', 'tarde', 'noche', 'madrugada']

    # Media y medianas globales por categoría, para referencia visual en los boxplots
    media_global = df[target].mean()
    mediana_global = df[target].median()

    # Bucle para generar los 8 gráficos
    for i, col in enumerate(columnas):
        if col == 'mes_deteccion':
            df[col] = pd.Categorical(df[col], categories=orden_meses, ordered=True)
        elif col == 'hora_deteccion':
            df[col] = pd.Categorical(df[col], categories=orden_hora, ordered=True)


        # Creamos el boxplot
        sns.boxplot(data=df,
                    x=col,
                    y=target,
                    ax=axes[i],
                    palette='viridis',
                    hue=col,
                    legend=False,
                    dodge=False,
                    width=0.6,
                    flierprops={'marker': 'o', 'markersize': 2,'alpha': 0.3})
        
        # Escala logarítmica en el eje Y
        axes[i].set_yscale('log')
        
        # Línea horizontal de la media y mediana global
        axes[i].axhline(media_global, color='red', linestyle='--', linewidth=1.5, label=f'Media: {media_global:.2f}')
        axes[i].axhline(mediana_global, color='blue', linestyle='-', linewidth=1.5, label=f'Mediana: {mediana_global:.2f}')

        # Estética de cada subgráfico
        axes[i].set_title(col, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Hectáreas (Log)')
        axes[i].tick_params(axis='x', rotation=20)

        if i == 0:  # Solo añadimos la leyenda en el primer gráfico para evitar repeticiones
            axes[i].legend(loc='upper right', fontsize=10)


    plt.tight_layout(h_pad=4.0, w_pad=2.0)
    fig.subplots_adjust(top=0.88)

    # Guardado de la imagen
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')

"""
=============================================================================
Función para generar un mapa de calor de correlaciones de Pearson entre las variables numéricas.
=============================================================================
"""

def mapa_calor_correlaciones(df) -> None:
    '''
    Función para generar un mapa de calor (heatmap) de la matriz de correlaciones de Pearson de las variables numéricas del DataFrame.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene las variables numéricas a analizar
    Output:
        Guarda una imagen PNG con el mapa de calor en la carpeta output.
    '''
    # Seleccion columnas numéricas
    columnas_numericas = df.select_dtypes(include=['number']).columns

    # Matriz de correlaciones de Pearson
    matriz_correlaciones = df[columnas_numericas].corr(method='pearson')

    # Mascara para ocultar la diagonal y los valores por encima de ella (redundantes)
    mascara = np.triu(np.ones_like(matriz_correlaciones, dtype=bool))
    plt.figure(figsize=(10, 8))
    plt.title('Matriz de Correlación de Pearson', fontsize=18, fontweight='bold', pad=20)
    
    # Dibujar el mapa de calor
    sns.heatmap(
        matriz_correlaciones, 
        mask=mascara,               # Aplicamos la máscara creada
        annot=True,                 # Mostramos los valores numéricos
        fmt=".2f",                  # Redondeamos a 2 decimales
        cmap='coolwarm',            # Paleta de colores divergente (azul=negativo, rojo=positivo)
        vmin=-1, vmax=1,            # Fijamos los límites matemáticos de Pearson
        center=0,                   # Centramos el color blanco en el cero
        square=True,                # Forzamos que las celdas sean cuadradas
        linewidths=0.5,             # Líneas finas separando las celdas
        cbar_kws={"shrink": 0.8}    # Reducimos un poco el tamaño de la barra lateral
    )
    
    # Estética de los ejes y etiquetas
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()

    # Guardado de la imagen
    ruta_guardado = 'practica_final_izquierdo_garcia_david/output/ej1_heatmap_correlaciones.png'
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')


"""
=============================================================================
Funcion para generar ungráfico de barras con la frecuencia relativa de cada categoría de una variable categórica dada.
=============================================================================
"""
def graficar_frecuencias_categoricas(df) -> None:
    '''
    Genera y guarda un panel con gráficos de barras horizontales mostrando la 
    frecuencia relativa (%) de cada categoría.
    Ordena de mayor a menor, excepto para variables temporales (orden cronológico).
    
    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
    Output:
        Guarda una imagen PNG con los gráficos en la carpeta output.
    '''

    # Seleccionamos solo las columnas categóricas
    columnas_cats = df.select_dtypes(include=['object', 'category']).columns

    # Cuadrícula de 4 filas y 2 columnas para 8 gráficos
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(20, 30))
    axes = axes.flatten()

    fig.suptitle('Frecuencia Relativa de Variables Categóricas (%)', fontsize=22, fontweight='bold')

    # Listas de orden para variables temporales
    orden_meses = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    orden_hora = ['manana', 'tarde', 'noche', 'madrugada']

    # Bucle para generar cada subplot
    for i, col in enumerate(columnas_cats):
        
        # Calcular frecuencias relativas en porcentaje
        frecuencias = df[col].value_counts(normalize=True) * 100
        
        # Aplicar orden cronológico si corresponde
        if col == 'mes_deteccion':
            orden_actual = [m for m in orden_meses if m in frecuencias.index]
            frecuencias = frecuencias.reindex(orden_actual)
        elif col == 'hora_deteccion':
            orden_actual = [h for h in orden_hora if h in frecuencias.index]
            frecuencias = frecuencias.reindex(orden_actual)
        else:
            frecuencias = frecuencias.sort_values(ascending=False)

        # Valores para el gráfico
        x_vals = frecuencias.values
        y_vals = frecuencias.index.astype(str).tolist()

        # Gráfico de barras horizontales
        sns.barplot(
            x=x_vals, 
            y=y_vals, 
            ax=axes[i], 
            palette='viridis',
            hue=y_vals,
            legend=False,
            dodge=False
        )
        
        # Estética y etiquetas
        axes[i].set_title(col, fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Porcentaje (%)')
        axes[i].set_ylabel('')
        
        # Añadir el número exacto al final de cada barra
        xmax = x_vals.max() if len(x_vals) > 0 else 100
        
        for p in axes[i].patches:
            width = p.get_width()
            if width > 0: # Solo si la barra existe
                # Por defecto: texto blanco, alineado a la derecha, justo antes del final de la barra
                x_pos = width - (xmax * 0.02)
                ha_align = 'right'
                color_texto = 'white'
                
                # Excepción de legibilidad: Si la barra es minúscula, 
                # el texto blanco no cabe dentro. Lo sacamos fuera en negro.
                if width < (xmax * 0.08): 
                    x_pos = width + (xmax * 0.01)
                    ha_align = 'left'
                    color_texto = 'black'

                axes[i].text(
                    x_pos, 
                    p.get_y() + p.get_height() / 2, # Posición Y centrada en la barra
                    f'{width:.1f}%', 
                    ha=ha_align, 
                    va='center', 
                    color=color_texto, 
                    fontweight='bold', 
                    fontsize=10
                )

    # Ajustar márgenes
    plt.tight_layout(h_pad=3.0, w_pad=2.0)
    fig.subplots_adjust(top=0.92)

    # Guardar el gráfico
    ruta_guardado = 'practica_final_izquierdo_garcia_david/output/ej1_categoricas.png'
    plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')


def graficar_scatter_relaciones(df) -> None:
    '''
    Genera gráficos de dispersión (scatter plots) para evaluar posibles relaciones 
    no lineales entre las variables numéricas predictoras y la variable objetivo.
    '''
    target = 'perdidas_superficie_ha'
    
    # 1. Seleccionamos las columnas numéricas excluyendo la variable objetivo
    columnas_numericas = df.select_dtypes(include=['number']).columns
    predictoras = [col for col in columnas_numericas if col != target]
    
    # 2. Configurar la cuadrícula (tenemos 5 predictoras, usaremos 2 filas x 3 columnas)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()

    fig.suptitle('Relación entre Variables Numéricas y Superficie Quemada', fontsize=20, fontweight='bold')

    # 3. Bucle para generar cada scatter plot
    for i, col in enumerate(predictoras):
        
        # Creamos el scatter plot con altísima transparencia (alpha) para evitar overplotting
        sns.scatterplot(
            data=df, 
            x=target, 
            y=col, 
            ax=axes[i], 
            alpha=0.05,        # <--- TRUCO: Transparencia para ver densidades
            color='steelblue',
            edgecolor=None     # Quitamos el borde del punto para que sea más limpio
        )
        
        # TRUCO: Usamos symlog (Symmetrical Log) para manejar la asimetría 
        # y que soporte los valores 0 sin dar error matemático.
        axes[i].set_xscale('log')
        
        # Estética
        axes[i].set_title(f'{col} vs Superficie', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Superficie Quemada', fontsize=12)
        axes[i].set_ylabel(col, fontsize=12)

    # 4. Eliminar el último subplot que queda vacío (5 predictoras en 6 huecos)
    fig.delaxes(axes[-1])

    # 5. Ajustar márgenes y guardar
    plt.tight_layout(h_pad=3.0, w_pad=2.0)
    fig.subplots_adjust(top=0.90)

    plt.show()

"""
=============================================================================
main del archivo, donde se ejecuta el análisis descriptivo completo.
=============================================================================
"""

if __name__ == "__main__":

    # Carga de datos
    df = pd.read_parquet('practica_final_izquierdo_garcia_david/data/incendios_limpio.parquet')

    # # verificación de la carga correcta
    # print(df.head())

    # # Tamaño del dataset
    # filas, columnas = df.shape
    # print("=== Resumen Estructural ===")
    # print(f"Número de filas: {filas}")
    # print(f"Número de columnas: {columnas}")
    # print(f"Tamaño en memoria: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    # print("========================================")

    # # Tipo de dato de cada columna
    # print("=== Tipos de Dato (dtypes) ===")
    # print(df.dtypes)
    # print("========================================")

    # # Porcentaje de valores nulos por columna
    # print("=== Porcentaje de Valores Nulos por Columna ===")
    # nulos = df.isnull().mean() * 100
    # print(nulos)
    # print("========================================")

    # # Estadísticos descriptivos de variables numéricas
    # print("=== Estadísticos Descriptivos de Variables Numéricas ===")
    # df.describe().to_csv('practica_final_izquierdo_garcia_david/output/ej1_descriptivo.csv', index=True)
    # print(df.describe())
    # print("========================================")

    # # Rango intercuartílico (IQR) de la variable objetivo: perdidas_superficie_ha
    # Q1 = df['perdidas_superficie_ha'].quantile(0.25)
    # Q3 = df['perdidas_superficie_ha'].quantile(0.75)
    # IQR = Q3 - Q1
    # print("=== Rango Intercuartílico (IQR) de la variable objetivo 'perdidas_superficie_ha' ===")
    # print(f"Q1: {Q1}")
    # print(f"Q3: {Q3}")
    # print(f"IQR: {IQR}")
    # print("========================================")

    # # Coeficiente de asimetría (skewness) y curtosis para las variables numéricas
    # print("=== Coeficiente de Asimetría (Skewness) y Curtosis ===")
    # columnas_numericas = df.select_dtypes(include=[np.number]).columns
    # skewness = df[columnas_numericas].skew()    
    # kurtosis = df[columnas_numericas].kurtosis()
    # skew_kurt_df = pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurtosis})
    # print(skew_kurt_df.round(2))
    # print("========================================")   

    # Generación de histogramas con tendencia KDE para las variables numéricas
    histogramas_variables_numericas(df)

    # Generación de boxplots de la variable objetivo por cada variable categórica
    columnas_g1 = ['hora_deteccion','clase_dia', 'tipo_ataque', 'intencionalidad']
    columnas_g2 = ['lugar_inicio', 'tipo_combustible', 'tipo_fuego', 'mes_deteccion']

    boxplot_variable_objetivo_por_categoria(df, columnas_g1, 'practica_final_izquierdo_garcia_david/output/ej1_boxplots_1.png', x_size=20)
    boxplot_variable_objetivo_por_categoria(df, columnas_g2, 'practica_final_izquierdo_garcia_david/output/ej1_boxplots_2.png', x_size=20)



    # # Detección de outliers usando el método IQR para la variable objetivo

    # outliers = df[(df['perdidas_superficie_ha'] < (Q1 - 1.5 * IQR)) | (df['perdidas_superficie_ha'] > (Q3 + 1.5 * IQR))]
    # print("=== Detección de Outliers usando el método IQR para 'perdidas_superficie_ha' ===")
    # print(f"Número de outliers detectados: {outliers.shape[0]}") 
    # print("========================================")

    # # Eliminación de outliers usando el método IQR para la variable objetivo
    # df_sin_outliers = df[~((df['perdidas_superficie_ha'] < (Q1 - 1.5 * IQR)) | (df['perdidas_superficie_ha'] > (Q3 + 1.5 * IQR)))]

# Generación del mapa de calor de correlaciones de Pearson entre las variables numéricas
    mapa_calor_correlaciones(df)

# Generación de gráficos de barras con la frecuencia relativa de cada categoría de las variables categóricas
    graficar_frecuencias_categoricas(df)

# Generación de gráficos de dispersión para evaluar posibles relaciones no lineales entre las variables numéricas predictoras y la variable objetivo
    #graficar_scatter_relaciones(df_sin_outliers)