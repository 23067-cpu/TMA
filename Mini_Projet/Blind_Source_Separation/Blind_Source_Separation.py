# ================================================================
# PROJET DSP - BLIND SOURCE SEPARATION (COCKTAIL PARTY PROBLEM)
# ================================================================
# Auteur : Projet académique
# Environnement recommandé : Google Colab
# Bibliothèques principales : numpy, matplotlib, scipy, sklearn
#
# Objectif :
# Séparer deux sources audio mélangées en utilisant l'algorithme
# FastICA (Independent Component Analysis).
#
# ================================================================


# ================================================================
# 1) INSTALLATION DES BIBLIOTHÈQUES (si nécessaire dans Colab)
# ================================================================

!pip install librosa soundfile scikit-learn


# ================================================================
# 2) IMPORTATION DES LIBRAIRIES
# ================================================================

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf

from sklearn.decomposition import FastICA
from scipy import signal

from IPython.display import Audio, display
from google.colab import files


# ================================================================
# 3) FONCTIONS UTILITAIRES
# ================================================================

def normaliser(signal_audio):
    """
    Cette fonction normalise un signal audio afin d'éviter
    le clipping lors de la lecture ou sauvegarde.
    """
    max_val = np.max(np.abs(signal_audio))
    
    if max_val == 0:
        return signal_audio
    
    return signal_audio / max_val


def afficher_waveform(signal_audio, sr, titre):
    """
    Affiche la forme d'onde d'un signal audio.
    """
    plt.figure(figsize=(12,4))
    temps = np.arange(len(signal_audio)) / sr
    
    plt.plot(temps, signal_audio)
    plt.title(titre)
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Amplitude")
    
    plt.show()


def afficher_spectrogramme(signal_audio, sr, titre):
    """
    Affiche le spectrogramme d'un signal audio.
    """
    
    plt.figure(figsize=(12,4))
    
    S = librosa.stft(signal_audio)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    
    librosa.display.specshow(S_db,
                             sr=sr,
                             x_axis='time',
                             y_axis='hz')
    
    plt.colorbar()
    plt.title(titre)
    
    plt.show()


def calculer_snr(reference, estimation):
    """
    Calcule le SNR (Signal to Noise Ratio).
    """
    
    longueur = min(len(reference), len(estimation))
    
    reference = reference[:longueur]
    estimation = estimation[:longueur]
    
    bruit = reference - estimation
    
    puissance_signal = np.sum(reference**2)
    puissance_bruit = np.sum(bruit**2)
    
    if puissance_bruit == 0:
        return np.inf
    
    snr = 10 * np.log10(puissance_signal / puissance_bruit)
    
    return snr


# ================================================================
# 4) CHARGEMENT DES FICHIERS AUDIO
# ================================================================

print("Veuillez uploader deux fichiers audio (WAV recommandé).")

uploaded = files.upload()

fichiers = list(uploaded.keys())

if len(fichiers) < 2:
    raise Exception("Veuillez uploader au moins deux fichiers audio.")


fichier1 = fichiers[0]
fichier2 = fichiers[1]


# Chargement avec librosa
audio1, sr = librosa.load(fichier1, sr=16000)
audio2, sr = librosa.load(fichier2, sr=16000)


# On ajuste les longueurs
longueur = min(len(audio1), len(audio2))

audio1 = audio1[:longueur]
audio2 = audio2[:longueur]


audio1 = normaliser(audio1)
audio2 = normaliser(audio2)


print("Sources originales chargées.")


# ================================================================
# 5) VISUALISATION DES SOURCES ORIGINALES
# ================================================================

afficher_waveform(audio1, sr, "Source 1 - Forme d'onde")
afficher_waveform(audio2, sr, "Source 2 - Forme d'onde")

afficher_spectrogramme(audio1, sr, "Spectrogramme Source 1")
afficher_spectrogramme(audio2, sr, "Spectrogramme Source 2")


print("Lecture des sources originales")

display(Audio(audio1, rate=sr))
display(Audio(audio2, rate=sr))


# ================================================================
# 6) CREATION DU MELANGE (MIXING)
# ================================================================

"""
Nous allons simuler deux microphones.

Chaque microphone capte une combinaison linéaire
des deux sources.
"""

A = np.array([
    [1.0, 0.6],
    [0.4, 1.0]
])

print("Matrice de mélange :")
print(A)


sources = np.vstack((audio1, audio2))

melanges = A @ sources


micro1 = melanges[0]
micro2 = melanges[1]


micro1 = normaliser(micro1)
micro2 = normaliser(micro2)


print("Lecture des signaux mélangés")

display(Audio(micro1, rate=sr))
display(Audio(micro2, rate=sr))


# ================================================================
# 7) APPLICATION DE L'ALGORITHME FASTICA
# ================================================================

"""
FastICA permet d'estimer les sources indépendantes
à partir des signaux observés.
"""

X = melanges.T

ica = FastICA(n_components=2,
              random_state=0,
              max_iter=500)

sources_estimees = ica.fit_transform(X)

sources_estimees = sources_estimees.T


source_est1 = normaliser(sources_estimees[0])
source_est2 = normaliser(sources_estimees[1])


print("Séparation terminée.")


# ================================================================
# 8) VISUALISATION DES SOURCES ESTIMEES
# ================================================================

afficher_waveform(source_est1, sr, "Source estimée 1")
afficher_waveform(source_est2, sr, "Source estimée 2")

afficher_spectrogramme(source_est1, sr, "Spectrogramme Source estimée 1")
afficher_spectrogramme(source_est2, sr, "Spectrogramme Source estimée 2")


print("Lecture des sources séparées")

display(Audio(source_est1, rate=sr))
display(Audio(source_est2, rate=sr))


# ================================================================
# 9) CALCUL DU SNR
# ================================================================

snr1 = calculer_snr(audio1, source_est1)
snr2 = calculer_snr(audio2, source_est2)

print("SNR Source 1 :", snr1, "dB")
print("SNR Source 2 :", snr2, "dB")


# ================================================================
# 10) SAUVEGARDE DES RESULTATS
# ================================================================

sf.write("source_separee_1.wav", source_est1, sr)
sf.write("source_separee_2.wav", source_est2, sr)

print("Fichiers sauvegardés.")


files.download("source_separee_1.wav")
files.download("source_separee_2.wav")


# ================================================================
# FIN DU PROJET
# ================================================================

print("Projet terminé avec succès.")