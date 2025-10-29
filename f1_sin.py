import numpy as np
import matplotlib.pyplot as plt
import os
from utils import generate_points, interpolate, mse, h1, h3, h4

# Parametry
N = 100
x_min, x_max = -np.pi, np.pi
multipliers = [2, 4, 10]
kernels = [h1, h3, h4]
kernel_names = ["h1", "h3", "h4"]

# Folder na wykresy (na razie nieużywany)
# output_dir = r"C:\Users\wloda\Desktop\sioc\sioc\images"
# os.makedirs(output_dir, exist_ok=True)

# Funkcja do interpolacji
def f1(x):
    return np.sin(x)

# Generowanie punktów oryginalnych
x_orig = generate_points(N, x_min, x_max, distribution="uniform")
y_orig = f1(x_orig)

# 🔹 Iteracja po jądrach konwolucji
for kernel, kname in zip(kernels, kernel_names):
    plt.figure(figsize=(12, 6))

    # Rysowanie oryginalnych punktów
    plt.plot(x_orig, y_orig, 'o', label='Oryginalne punkty', markersize=5, color='black')

    # 🔹 Interpolacja dla różnych multiplikatorów
    for multiplier in multipliers:
        x_new = np.linspace(x_min, x_max, N * multiplier)
        y_new = interpolate(x_orig, y_orig, x_new, kernel)
        y_true = f1(x_new)
        error = mse(y_true, y_new)

        # Wypisanie błędu
        print(f"Jądro: {kname}, multiplikator: {multiplier}x, MSE: {error:.6f}")

        # Rysowanie interpolacji
        plt.plot(x_new, y_new, label=f'{multiplier}x (MSE={error:.6f})')

    # 🔹 Formatowanie wykresu
    plt.title(f'Interpolacja f1(x) = sin(x) z użyciem jądra {kname}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

    # 🔹 Zapis wykresu (na razie wyłączony)
    # save_path = os.path.join(output_dir, f"f1_{kname}.png")
    # plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.close()

    # 🔹 Pokazanie wykresu (aktywnie działa)
    plt.show()

    print(f"✅ Wykres dla jądra {kname} został wygenerowany.")
