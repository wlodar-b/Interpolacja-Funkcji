import numpy as np

# Funkcje pomocnicze

def generate_points(N, x_min, x_max, distribution="uniform"):
    """
    Generuje N punktów w przedziale [x_min, x_max].
    distribution:
        "uniform"  - równomiernie
        "normal"   - rozkład normalny
        "nonuniform" - gęstsze punkty w centrum przedziału
    """
    if distribution == "uniform":
        return np.linspace(x_min, x_max, N)
    elif distribution == "normal":
        mu = (x_max + x_min) / 2
        sigma = (x_max - x_min) / 6  # większość punktów w [x_min, x_max]
        points = np.random.normal(mu, sigma, N)
        return np.clip(points, x_min, x_max)
    elif distribution == "nonuniform":
        # przykładowy nierównomierny rozkład: więcej punktów w centrum
        t = np.linspace(0, 1, N)
        t = 0.5 - 0.5 * np.cos(np.pi * t)  # gęstsze punkty w środku (rozkład cosinusowy)
        return x_min + t * (x_max - x_min)
    else:
        raise ValueError("Nieobsługiwany typ rozkładu")

# Jądra konwolucji

def h1(t):
    """Prostokątne: 1 dla t ∈ [0,1), 0 w pozostałych"""
    return np.where((t >= 0) & (t < 1), 1.0, 0.0)

def h2(t):
    """Prostokątne: 1 dla t ∈ [-0.5, 0.5), 0 w pozostałych"""
    return np.where((t >= -0.5) & (t < 0.5), 1.0, 0.0)

def h3(t):
    """Trójkątne: 1-|t| dla |t|<=1, 0 w pozostałych"""
    return np.where(np.abs(t) <= 1, 1 - np.abs(t), 0.0)

def h4(t, a=4):
    """
    Sinc z ograniczeniem dziedziny do [-a, a]
    np.sinc(x) = sin(pi*x)/(pi*x)
    """
    t = np.array(t)
    y = np.sinc(t / np.pi)
    y[np.abs(t) > a] = 0
    return y

# Funkcja interpolacji 1D

def interpolate(x, y, x_new, kernel):
    """
    Interpolacja 1D przez konwolucję.
    x, y       - oryginalne punkty
    x_new      - punkty, które chcemy wygenerować
    kernel     - funkcja jądra konwolucji
    """
    y_new = np.zeros_like(x_new)
    dx = x[1] - x[0]  # krok oryginalnej siatki
    
    for i, xi in enumerate(x_new):
        t = (xi - x) / dx
        weights = kernel(t)
        if np.sum(weights) != 0:
            y_new[i] = np.sum(weights * y) / np.sum(weights)
        else:
            y_new[i] = 0
    return y_new

# Funkcja do obliczenia MSE

def mse(y_true, y_pred):
    """Oblicza średni kwadratowy błąd między dwoma wektorami"""
    return np.mean((y_true - y_pred)**2)
