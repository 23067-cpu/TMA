"""
TP3 - Aliasing (sous-échantillonnage) et Quantification d'image (Colab ready)

Partie Audio :
- Génération d'un signal test (44.1 kHz) puis sous-échantillonnage par 10 -> aliasing
- Affichage des spectrogrammes et écoute (si Colab)

Partie Image :
- Génération d'une image dégradée (gradient) si absence d'image fournie
- Réduction de niveaux de gris à 4 et 2 niveaux
- Pixelisation (réduction de résolution puis upscale) pour montrer effet

Instructions :
- Si vous avez un fichier audio 'input.wav' ou une image 'input.png' dans le répertoire, le script les utilisera.
- Sinon, le script génèrera des exemples synthétiques.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import decimate
from scipy.io import wavfile
from PIL import Image

def partie_audio(fs=44100, duration=2.0):
    t = np.arange(0, duration, 1/fs)
    # signal large bande : somme de sinusoides pour simuler riches harmoniques
    sig = 0.5*np.sin(2*np.pi*500*t) + 0.3*np.sin(2*np.pi*3000*t) + 0.2*np.sin(2*np.pi*7000*t)
    # sauvegarder test si aucun fichier fourni
    wavfile.write("tp3_input_generated.wav", fs, np.int16(sig/np.max(np.abs(sig)) * 32767))
    print("Fichier audio généré : tp3_input_generated.wav")

    # sous-échantillonnage sauvage : ne garder qu'un échantillon sur 10
    factor = 10
    sig_sub = sig[::factor]
    fs_sub = fs // factor

    # tracés temporels
    plt.figure(figsize=(10,3))
    plt.plot(t[:2000], sig[:2000])
    plt.title("Signal original (début)")
    plt.xlabel("Temps (s)")
    plt.tight_layout()
    plt.show()

    t_sub = np.arange(0, len(sig_sub))/fs_sub
    plt.figure(figsize=(10,3))
    plt.plot(t_sub[:500], sig_sub[:500])
    plt.title(f"Signal sous-échantillonné par {factor} (début) - aliasing attendu")
    plt.xlabel("Temps (s)")
    plt.tight_layout()
    plt.show()

    # spectrogrammes
    plt.specgram(sig, Fs=fs)
    plt.title("Spectrogramme - signal original")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    plt.specgram(sig_sub, Fs=fs_sub)
    plt.title("Spectrogramme - signal sous-échantillonné (fs/10)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    wavfile.write("tp3_subsampled.wav", fs_sub, np.int16(sig_sub/np.max(np.abs(sig_sub)) * 32767))
    print("Fichier audio sous-échantillonné : tp3_subsampled.wav (écoute possible dans Colab)")

def partie_image():
    # tenter d'ouvrir input.png sinon générer un gradient synthétique
    try:
        img = Image.open("input.png").convert("L")
        print("Image 'input.png' trouvée et chargée.")
    except Exception as e:
        print("Aucune image fournie, génération d'un gradient synthétique.")
        W, H = 512, 256
        gradient = np.tile(np.linspace(0,255, W, dtype=np.uint8), (H,1))
        img = Image.fromarray(gradient, mode='L')
        img.save("tp3_generated_gradient.png")
        print("Image générée : tp3_generated_gradient.png")

    arr = np.array(img)
    plt.figure(figsize=(6,3))
    plt.imshow(arr, cmap='gray', vmin=0, vmax=255)
    plt.title("Image originale (niveau de gris)")
    plt.axis('off')
    plt.show()

    def quantize_levels(a, levels):
        # map to specified number of levels in [0,255]
        maxv = 255
        scaled = (a / maxv) * (levels-1)
        q = np.round(scaled) * (maxv/(levels-1))
        return q.astype(np.uint8)

    for lv in [4, 2]:
        q = quantize_levels(arr, lv)
        plt.figure(figsize=(6,3))
        plt.imshow(q, cmap='gray', vmin=0, vmax=255)
        plt.title(f"Quantification à {lv} niveaux (effet de banding/contouring)")
        plt.axis('off')
        plt.show()

    # pixelisation : réduire la résolution par 8 puis remettre à la taille originale
    small = img.resize((img.width//8, img.height//8), Image.NEAREST)
    pixelized = small.resize(img.size, Image.NEAREST)
    pixelized.save("tp3_pixelized.png")
    plt.figure(figsize=(6,3))
    plt.imshow(np.array(pixelized), cmap='gray', vmin=0, vmax=255)
    plt.title("Image pixelisée (réduction x8 puis upscale)")
    plt.axis('off')
    plt.show()
    print("Image pixelisée sauvegardée : tp3_pixelized.png")

if __name__ == "__main__":
    partie_audio()
    partie_image()
