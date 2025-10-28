import numpy as np
import matplotlib.pyplot as plt
from utils import generate_points, interpolate, mse, h1, h2, h3, h4

# Parametry

N = 100 # liczba punktów początkowych
x_min, x_max = -np.pi, np.pi
multipliers = [2, 4, 10] # ile razy zwiększamy liczbę punktów
kernels = [h1, h3, h4] # wybrane jądro konwolucji
kernel_names = ["h1", "h3", "h4"]

# Funkcje do interpolacji

def f3(x):
    return np.sign(np.sin(8 * x))

# Generowanie punktów oryginalnych

x_orig = generate_points(N, x_min, x_max, distribution="uniform")
y_orig = f3(x_orig)

# Wykres oryginalnych punktów

plt.figure(figsize=(12, 6))
plt.plot(x_orig, y_orig, 'o', label='Oryginalne punkty', markersize=5)

# Interpolacja

for multiplier in multipliers:
    x_new = np.linspace(x_min, x_max, N * multiplier)

    for kernel, kname in zip(kernels, kernel_names):
        y_new = interpolate(x_orig, y_orig, x_new, kernel)
        y_true = f3(x_new)
        error = mse(y_true, y_new)

        print(f"Multiplikator: {multiplier}x, Jądro: {kname}, MSE: {error:.6f}")

        plt.plot(x_new, y_new, label=f'{kname}, {multiplier}x')

# Wykończenie wykresu

plt.title('Interpolacja funkcji f3(x) = sgn(sin(8x))')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

