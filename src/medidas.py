import numpy as np
import pandas as pd


def media_evolve(lista_datos: list) -> float:
    return sum(lista_datos)/len(lista_datos)

def mediana_evolve(lista_datos: list) -> float:
    n = len(lista_datos)
    lista_ordenada = sorted(lista_datos)
    if n % 2 == 0:
        centro_der = n//2
        centro_izq = centro_der - 1
        return (lista_ordenada[centro_izq] + lista_ordenada[centro_der])/2
    else:
        return lista_ordenada[n//2]

def percentil_evolve(lista_datos: list, percentil: int) -> float:
    n = len(lista_datos)
    lista_ordenada = sorted(lista_datos)
    posicion = (n-1) * (percentil/100)
    indice_posicion = int(posicion)
    parte_decimal = posicion - indice_posicion
    if parte_decimal == 0 or indice_posicion == n-1:
        return lista_ordenada[indice_posicion]
    else:
        indice_siguiente = indice_posicion + 1
        return lista_ordenada[indice_posicion] + parte_decimal * (lista_ordenada[indice_siguiente] - lista_ordenada[indice_posicion])

def varianza_evolve(lista_datos: list) -> float:
    n = len(lista_datos)
    media = media_evolve(lista_datos)
    return sum((x - media) ** 2 for x in lista_datos) / (n-1)

def desviacion_evolve(lista_datos: list) -> float:
    return varianza_evolve(lista_datos) ** 0.5

def IQR_evolve(lista_datos: list) -> float:
    q1 = percentil_evolve(lista_datos, 25)
    q3 = percentil_evolve(lista_datos, 75)
    return q3 - q1

def skewness_evolve(lista_datos: list) -> float:
    media = media_evolve(lista_datos)
    desviacion = desviacion_evolve(lista_datos)
    n = len(lista_datos)
    return (sum((x - media) ** 3 for x in lista_datos))/ (n-1) * desviacion ** 3
    
def kurtosis_evolve(lista_datos: list) -> float:
    media = media_evolve(lista_datos)
    desviacion = desviacion_evolve(lista_datos)
    n = len(lista_datos)
    return (sum((x - media) ** 4 for x in lista_datos))/ (n) * desviacion ** 4



if __name__ == "__main__":
    
    np.random.seed(42)
    edad = list(np.random.randint(20, 60, 100))
    salario =  list(np.random.normal(45000, 15000, 100))
    experiencia = list(np.random.randint(0, 30, 100))


    np.random.seed(42)
    df = pd.DataFrame({
        'edad': np.random.randint(20, 60, 100),
        'salario': np.random.normal(45000, 15000, 100),
        'experiencia': np.random.randint(0, 30, 100)
    })

    print("resultado pandas")
    print("--------------------------------")
    print(df.describe())

    print("resultado funciones")
    print("--------------------------------")

    print("Medidas para la edad:")
    print(f"media: {media_evolve(edad):.2f}")
    print(f"mediana: {mediana_evolve(edad):.2f}")
    print(f"percentil 50: {percentil_evolve(edad, 50):.2f}")
    print(f"varianza: {varianza_evolve(edad):.2f}")
    print(f"desviacion estándar: {desviacion_evolve(edad):.2f}")
    print(f"IQR: {IQR_evolve(edad):.2f}")

    print("Medidas para el salario:")
    print(f"media: {media_evolve(salario):.2f}")
    print(f"mediana: {mediana_evolve(salario):.2f}")
    print(f"percentil 50: {percentil_evolve(salario, 50):.2f}")
    print(f"varianza: {varianza_evolve(salario):.2f}")
    print(f"desviacion estándar: {desviacion_evolve(salario):.2f}")
    print(f"IQR: {IQR_evolve(salario):.2f}")

    print("Medidas para la experiencia:")
    print(f"media: {media_evolve(experiencia):.2f}")
    print(f"mediana: {mediana_evolve(experiencia):.2f}")
    print(f"percentil 50: {percentil_evolve(experiencia, 50):.2f}")
    print(f"varianza: {varianza_evolve(experiencia):.2f}")
    print(f"desviacion estándar: {desviacion_evolve(experiencia):.2f}")
    print(f"IQR: {IQR_evolve(experiencia):.2f}")