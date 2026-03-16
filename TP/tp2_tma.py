"""
TP2 - Analyse spectrale et filtrage (ready for Colab)

Contenu :
1) Synthèse d'un signal audio (440 Hz et 880 Hz), FFT et tracé
2) Ajout d'un sifflement (5000 Hz), identification du pic et filtrage dans le domaine fréquentiel
3) Reconstruction (IFFT) et comparaison

Notes :
- Ce script sauvegarde un fichier WAV (signal_original.wav et signal_filtre.wav) dans le répertoire courant.
- Pour écouter dans Colab : from IPython.display import Audio; Audio("signal_filtre.wav")
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from scipy.io import wavfile

def synthese_tons(fs=44100, duration=2.0, f1=440, f2=880):
    t = np.arange(0, duration, 1/fs)
    sig = 0.6*np.sin(2*np.pi*f1*t) + 0.4*np.sin(2*np.pi*f2*t)
    return t, sig

def calcul_fft(sig, fs):
    N = len(sig)
    S = fft(sig)
    freqs = fftfreq(N, 1/fs)
    return freqs, S

def plot_fft_magnitude(freqs, S, xlim=None, title="Module de la FFT"):
    N = len(S)
    # on trace seulement la partie positive
    mask = freqs >= 0
    plt.figure(figsize=(10,4))
    plt.plot(freqs[mask], np.abs(S[mask]) / N)
    plt.title(title)
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Amplitude (normée)")
    if xlim:
        plt.xlim(xlim)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def ajouter_sifflement(t, sig, fbruit=5000, amp=0.15):
    bruit = amp * np.sin(2*np.pi*fbruit*t)
    return sig + bruit

def filtre_freq(S, freqs, f_center, bw=50, mode='notch'):
    # bw : demi-largeur autour de f_center en Hz (on annule les bins dans [f_center-bw, f_center+bw])
    S_f = S.copy()
    # traiter positifs et négatifs
    mask_pos = (np.abs(freqs - f_center) <= bw)
    mask_neg = (np.abs(freqs + f_center) <= bw)
    mask = mask_pos | mask_neg
    S_f[mask] = 0
    return S_f

def sauvegarder_wav(nom, sig, fs):
    # normaliser pour int16
    sig_norm = sig / np.max(np.abs(sig))
    data = np.int16(sig_norm * 32767)
    wavfile.write(nom, fs, data)
    print(f"Fichier sauvegardé : {nom}")

if __name__ == "__main__":
    fs = 44100
    t, sig = synthese_tons(fs=fs, duration=2.0, f1=440, f2=880)
    freqs, S = calcul_fft(sig, fs)
    plot_fft_magnitude(freqs, S, xlim=(0,5000), title="Spectre - signal pur")

    sig_bruit = ajouter_sifflement(t, sig, fbruit=5000, amp=0.12)
    freqs_b, S_b = calcul_fft(sig_bruit, fs)
    plot_fft_magnitude(freqs_b, S_b, xlim=(0,8000), title="Spectre - avec sifflement (5000 Hz)")

    # filtrage coupe-bande (notch) autour de 5000 Hz
    S_filtre = filtre_freq(S_b, freqs_b, f_center=5000, bw=30, mode='notch')
    sig_filtre = np.real(ifft(S_filtre))

    plot_fft_magnitude(freqs_b, S_filtre, xlim=(0,8000), title="Spectre - après filtrage (5000 Hz atténué)")

    # sauvegarder audio (original et filtré)
    sauvegarder_wav("tp2_signal_original.wav", sig_bruit, fs)
    sauvegarder_wav("tp2_signal_filtre.wav", sig_filtre, fs)

    # affichage temporel pour comparaison
    plt.figure(figsize=(10,3))
    plt.plot(t[:1000], sig_bruit[:1000], label='Signal bruité (début)')
    plt.plot(t[:1000], sig_filtre[:1000], label='Signal filtré (début)', alpha=0.8)
    plt.legend()
    plt.title("Comparaison temporelle (début du signal)")
    plt.xlabel("Temps (s)")
    plt.tight_layout()
    plt.show()

    print("Dans Colab, vous pouvez écouter :\nfrom IPython.display import Audio\nAudio('tp2_signal_original.wav')\nAudio('tp2_signal_filtre.wav')")
