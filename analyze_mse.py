import numpy as np 
import matplotlib.pyplot as plt
from utils import generate_points, interpolate, mse, h1, h3, h4

# Funkcje do interpolacji 

def f1(x):
    return np.sin(x)

def f2(x):
    return np.sin(1/x)

def f3(x):
    return np.sign(np.sin(8 * x))

# Parametry globalne

N = 100
x_min, x_max = -np.pi, np.pi
multipliers = [2, 3, 10]
kernels = [h1, h3, h4]
kernel_names = ["h1", "h3", "h4"]
functions = [f1, f2, f3]
function_names = ["sin(x)", "sin(1/x)", "sgn(sin(8x))"]

# Wyniki

results = []

for func, fname in zip(functions, function_names):
    print(f"\n=== Funkcja: {fname} ===")
    x_orig = generate_points(N, x_min, x_max, distribution="uniform")
    y_orig = func(x_orig)

    for kernel, kname in zip(kernels, kernel_names):
        for multiplier in multipliers:
            x_new = np.linspace(x_min, x_max, N * multiplier)
            y_true = func(x_new)
            y_new = interpolate(x_orig, y_orig, x_new, kernel)
            error = mse(y_true, y_new)

            results.append((fname, kname, multiplier, error))
            print(f"Jądro: {kname:>3} | Mnożnik: {multiplier:>2}x | MSE = {error:.6f}")

# Wykres MSE 

plt.figure(figsize=(10, 6))

for func_name in function_names:
    for kname in kernel_names:
        subset = [(mult, err) for (fname, kern, mult, err) in results if fname == func_name and kern == kname]
        if subset:
            x_vals, y_vals = zip(*subset)
            plt.plot(x_vals, y_vals, marker='o', label=f"{func_name} - {kname}")

plt.title("Porównanie jakości interpolacji (MSE)")
plt.xlabel("Mnożnik liczby punktów")
plt.ylabel("MSE (im mniejsze, tym lepiej)")
plt.legend()
plt.grid(True)
plt.show()