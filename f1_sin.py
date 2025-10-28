import numpy as np
import matplotlib.pyplot as plt
import os  # potrzebne do pracy ze ścieżkami
from utils import generate_points, interpolate, mse, h1, h3, h4

# Parametry
N = 100  # liczba punktów początkowych
x_min, x_max = -np.pi, np.pi
multipliers = [2, 4, 10]  # ile razy zwiększamy liczbę punktów
kernels = [h1, h3, h4]  # wybrane jądra konwolucji
kernel_names = ["h1", "h3", "h4"]

# 🔹 Ścieżka do folderu, w którym mają być zapisywane wykresy
output_dir = r"C:\Users\wloda\Desktop\sioc\sioc\images"
os.makedirs(output_dir, exist_ok=True)  # utworzy folder jeśli nie istnieje

# Funkcja do interpolacji
def f1(x):
    return np.sin(x)

# Generowanie punktów oryginalnych
x_orig = generate_points(N, x_min, x_max, distribution="uniform")
y_orig = f1(x_orig)

# Wykres oryginalnych punktów
plt.figure(figsize=(12, 6))
plt.plot(x_orig, y_orig, 'o', label='Oryginalne punkty', markersize=5)

# Interpolacja
for multiplier in multipliers:
    x_new = np.linspace(x_min, x_max, N * multiplier)

    for kernel, kname in zip(kernels, kernel_names):
        y_new = interpolate(x_orig, y_orig, x_new, kernel)
        y_true = f1(x_new)
        error = mse(y_true, y_new)

        print(f"Multiplikator: {multiplier}x, Jądro: {kname}, MSE: {error:.6f}")
        plt.plot(x_new, y_new, label=f'{kname}, {multiplier}x')

# Wykończenie wykresu
plt.title('Interpolacja funkcji f1(x) = sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# 🔹 Zapis wykresu do folderu
save_path = os.path.join(output_dir, "f1_plot.png")  # pełna ścieżka do pliku
plt.savefig(save_path, dpi=300, bbox_inches='tight')  # zapis z wysoką jakością

# Wyświetlenie wykresu
plt.show()

print(f"Wykres zapisano w: {save_path}")
