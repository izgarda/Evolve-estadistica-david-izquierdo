import math
import random
import matplotlib.pyplot as plt
import numpy as np




# Parameters:
a = 2.0
b = 1.0
xmin = 0.0
xmax = 10.0
noise = 2.0
n = 100

# Randomly generated problem data:
np.random.seed(42)
x = xmin + np.random.rand(n)*(xmax - xmin)
t = a*x + b + np.random.randn(n)*noise

    
def regresion_lineal_simple(x, t, num_iters=8, eta=0.01):
        w = np.random.randn()
        b = np.random.randn()

        xmin, xmax = x.min(), x.max()

        for i in range(num_iters):

            # Definición de la recta
            y = w * x + b

            #Cálculo del error
            error = y - t

            #Cálculo del gradiente de lo pesos b, w1
            dw = (2*np.sum(error*x))/len(x)
            db = (2*np.sum(error))/len(x)

            #Actualización de parametros
            w -= dw *eta
            b -= db * eta
            

            plt.figure(figsize=(6, 6))
            plt.plot(x, t, 'o', label='Datos reales')
            plt.plot([xmin, xmax], [w * xmin + b, w * xmax + b], 'r-', label=f'Modelo (iter {i+1})')
            plt.grid(True)
            plt.xlabel("x")
            plt.ylabel("t")
            plt.title(f"Iteración {i+1}")
            plt.legend()
            plt.show()

        return w, b

def show_data():

    plt.figure(figsize=(6, 6))
    plt.plot(x, t, 'o')
    plt.plot([xmin, xmax], [a*xmin + b, a*xmax + b], 'r-')
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

def main():
    show_data()


if "__name__" == "__main__":
    main()