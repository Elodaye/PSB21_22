import signal

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# !pip install PyWavelets
import pywt
import pywt.data
import soundfile
from scipy.io import wavfile


def wav_to_spect(wav_name,output_name, output_dir, expected_time):
    """
    Conversion d'un fichier .wav, à partir de son nom, en spectrogramme
    et enregistrement dans le dossier indiqué par le path sous le nom
    output_name
    """

    ### On récupère les données, qui sont toutes en .vaw et de 10 secondes
    sample_rate, samples = wavfile.read(wav_name)  # fréquence d'échantillonage //  Sample 1D si audio et 2D si stéréo
    time = samples.size / sample_rate

    if time == expected_time:  # si le fichier fait bien 10 secondes
        nperseg = 512 #4094*2
        nfft = nperseg  # i.e. pas de zero-padding

        ### Obtention du spectrogramme, frequencies est un array
        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=nperseg, nfft=nfft)
        ##spectrogram: fréquence en absisse, temps en ordonnées

        ### Coupure du spectrogramme (limitation des fréquences)
        fmin = 100
        fmax = 15000
        freqs_to_keep = (frequencies == frequencies)
        freqs_to_keep *= fmin <= frequencies
        freqs_to_keep *= frequencies <= fmax
        spectrogram = spectrogram[freqs_to_keep, :]
        frequencies = frequencies[freqs_to_keep]
        #print("hu_spectrogram", spectrogram, spectrogram.shape)
        human_spectrogram =200* np.log(spectrogram)
        for i, ligne in enumerate (human_spectrogram):
            for j, colonne in enumerate(ligne) :
                if colonne <200:
                    human_spectrogram[i][j] = 200
                    #human_spectrogram = 100 * np.log10(spectrogram)
        #human_spectrogram = spectrogram
        # la plupart des fréquences sont basses, donc pour que les différentes classes soient différenciées plus clairement,
        # étale les faible fréquences, avec le log

        ### Création de la figure
        t_max = times[-1]
        t_ref_normalize = int(t_max / 10) + 1  # Taille de l'image proportionelle au temps
        fig = plt.figure(frameon=False, figsize=(5 * t_ref_normalize, 5))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        plt.pcolormesh(times, frequencies, human_spectrogram , cmap='binary') ##gray_r
        #plt.colorbar()
        #output = os.path.join(output_dir,output_name)
        output = output_dir + "//" +  output_name
        plt.savefig(output, dpi=fig.dpi)
        #plt.show()
        plt.close(fig)
    else:
        print("Fichier audio de mauvaise longueur! Attendu: ", expected_time, "| Reçu: ", time)

if __name__ == '__main__':

