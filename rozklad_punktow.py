import numpy as np
import matplotlib.pyplot as plt 
from scipy.interpolate import CubicSpline
from sklearn.metrics import mean_squared_error
import os

# Funkcja oryginalna
def f(x):
    return np.sin(x)

# Konfiguracja
N = 15
x_min, x_max = 0, 10
x_test = np.linspace(x_min, x_max, 500)
y_true = f(x_test)

# Przygotowanie folderu na wykresy
os.makedirs("plots", exist_ok=True)

# Zestawy punktów
x_even = np.linspace(x_min, x_max, N)
x_uniform = np.random.uniform(x_min, x_max, N)
x_normal = np.random.normal(loc=(x_min+x_max)/2, scale=(x_max-x_min)/5, size=N)

# Posortuj punkty, żeby interpolacja działała poprawnie
x_uniform.sort()
x_normal.sort()

# Interpolacja
datasets = {
    "Rownomierne": x_even,
    "Losowe (jednostajny)": x_uniform,
    "Losowe (normalny)": x_normal
}


results = {}

for name, x_points in datasets.items():
    y_points = f(x_points)

    # Interpolacja splajnami kubicznymi
    cs = CubicSpline(x_points, y_points)
    y_interp = cs(x_test)

    # Oblicz błąd MSE
    mse = mean_squared_error(y_true, y_interp)
    results[name] = mse

    # Wykres
    plt.figure(figsize=(8,5))
    plt.plot(x_test, y_true, 'k--', label='Funkcja oryginalna')
    plt.plot(x_test, y_interp, 'b', label='Interpolacja')
    plt.scatter(x_points, y_points, color='red', label='Punkty węzłowe')
    plt.title(f'Interpolacja ({name})\nMSE = {mse:.6f}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"plots/interpolacja_{name.replace(' ', '_')}.png")
    plt.close()


# --- Wyniki ---
print("Porównanie błędów MSE dla różnych rozmieszczeń punktów:")
for name, mse in results.items():
    print(f"{name}: {mse:.6f}")

# --- Zapis wyników do pliku tekstowego ---
with open("plots/wyniki_rozkladow.txt", "w") as f_out:
    f_out.write("Porównanie błędów MSE dla różnych rozmieszczeń punktów:\n")
    for name, mse in results.items():
        f_out.write(f"{name}: {mse:.6f}\n")

print("\nWyniki i wykresy zapisane w folderze 'plots/'.")