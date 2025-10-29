import numpy as np
import matplotlib.pyplot as plt
from utils import generate_points, interpolate, mse, h1, h3, h4

# Funkcje testowe
def f1(x):
    return np.sin(x)

def f2(x):
    return np.sin(1/x)

def f3(x):
    return np.sign(np.sin(8 * x))

# Parametry 
x_min, x_max = -np.pi, np.pi
Ns = [50, 100, 200, 500] # liczba punktów
distributions = ["uniform", "nonuniform"] # rozmieszczenie
kernels = [h1, h3, h4]
kernel_names = ["h1", "h3", "h4"]
functions = [f1, f2, f3]
function_names = ["sin(x)", "sin(1/x)", "sgn(sin(8x))"]

results = []

print("\n=== Analiza wpływu liczby i rozmieszczenia punktów na MSE ===\n")

# Główna pętla
for func, fname in zip(functions, function_names):
    print(f"\n### Funkcja: {fname} ###")
    for dist in distributions:
        print(f"\n Rozkład: {dist}")
        for N in Ns:
            x_orig = generate_points(N, x_min, x_max, distribution=dist)
            y_orig = func(x_orig)

            x_new = np.linspace(x_min, x_max, 1000)
            y_true = func(x_new)

            for kernel, kname in zip(kernels, kernel_names):
                y_new = interpolate(x_orig, y_orig, x_new, kernel)
                error = mse(y_true, y_new)
                results.append((fname, dist, N, kname, error))
                print(f"    N={N:>4} | Jądro={kname:<3} | MSE={error:.6f}")


# Wizualizacja wyników
plt.figure(figsize=(12, 7))

for func_name in function_names:
    for dist in distributions:
        for kname in kernel_names:
            subset = [(N, err) for (fname, d, N, kern,  err) in results if fname == func_name and d == dist and kern == kname]
            if subset:
                x_vals, y_vals = zip(*subset)
                plt.plot(x_vals, y_vals, marker="o", label=f"{func_name}, {dist}, {kname}")

plt.title("Wpływ liczby i rozmieszczenia punktów na jakość konwolucji (MSE)")
plt.xlabel("Liczba punktów N")
plt.ylabel("MSE (im mniejsze, tym lepiej)")
plt.legend()
plt.grid(True)
plt.show()