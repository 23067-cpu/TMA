"""
TP1 - Technologies Multimédias (TP_2026_TMA)
Script prêt pour Google Colab / exécution locale.

Contenu :
1) Génération d'un signal sinusoïdal
2) Ajout de bruit (BBG) et visualisation
3) Création d'un signal porte (rect) et convolution

Instructions :
- Ouvrir ce fichier dans Colab (Files > Upload) ou copier-coller dans une cellule.
- Exécuter la cellule. Les figures seront affichées inline.
- Aucune dépendance externe autre que numpy & matplotlib (fourni dans Colab).

Auteur : Assistant
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def tp1_sinusoidal():
    # paramètres
    f0 = 10.0        # Hz
    A = 1.0
    phi = 0.0
    fs = 100        # Hz
    duration = 1.0  # seconde

    t = np.arange(0, duration, 1/fs)
    x = A * np.sin(2 * np.pi * f0 * t + phi)

    # tracé
    plt.figure(figsize=(10,3))
    plt.plot(t, x)
    plt.title("Signal sinusoïdal x(t) = A sin(2π f0 t + φ)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return t, x

def tp1_noise(t, x, snr_db=20):
    # Génération de Bruit Blanc Gaussien (BBG)
    rng = np.random.default_rng(0)
    # calculer puissance signal et bruit pour un SNR donné
    ps = np.mean(x**2)
    snr_linear = 10**(snr_db/10)
    pn = ps / snr_linear
    sigma = np.sqrt(pn)
    bruit = rng.normal(0, sigma, size=x.shape)
    y = x + bruit

    # tracé comparatif
    plt.figure(figsize=(10,4))
    plt.plot(t, x, label='Signal pur')
    plt.plot(t, y, label=f'Signal bruité (SNR={snr_db} dB)', alpha=0.7)
    plt.title("Signal pur vs signal bruité")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return y, bruit

def tp1_rect_convolution():
    N = 100
    rect = np.zeros(N)
    rect[20:41] = 1.0

    conv = np.convolve(rect, rect, mode='full')

    # Signal porte
    plt.figure(figsize=(10,3))
    plt.stem(np.arange(N), rect, basefmt=" ")
    plt.title("Signal porte (rect)")
    plt.xlabel("Indice n")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    # Convolution
    plt.figure(figsize=(10,3))
    plt.stem(np.arange(len(conv)), conv, basefmt=" ")
    plt.title("Convolution rect * rect")
    plt.xlabel("Indice n")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    return rect, conv

if __name__ == "__main__":
    t, x = tp1_sinusoidal()
    y, bruit = tp1_noise(t, x, snr_db=15)
    rect, conv = tp1_rect_convolution()
