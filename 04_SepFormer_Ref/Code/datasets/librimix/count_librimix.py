import os
import numpy as np
from scipy.io import wavfile

# Path to the LibriMix folder and selected subfolder
librimix_path = '/isip/Students-ro/SpeechSeparation/LibriMix/LibriMix'


def calculate_hours(sampling_length, librimix='Libri2Mix', sampling_rate=8000):
    """ # TODO Beschreibung
        :param librimix: 'Libri2Mix', 'Libri3Mix
        :param sampling_rate: '8000', '16000'
        :param sampling_length: 'min', 'max'
    """
    # Parameters
    wav_type = "wav{}k".format(sampling_rate // 1000)
    subfolder = ['mix_both']  # ['mix_both', 'mix_clean', 'mix_single', 'noise', 's1', 's2']
    dataset = ['dev', 'test', 'train-100', 'train-360']

    # Path to the subfolder containing the different types of data for the network
    folder_path = os.path.join(librimix_path, librimix, wav_type, sampling_length)

    # Berechne die Anzahl an Werten bzw. Stunden der gesamten Audiolänge
    for single_dataset in dataset:
        number_of_entries = 0
        number_of_files = 0
        # Subfolder with files of sounds (just for 'mix_both')
        subdir_path = os.path.join(folder_path, single_dataset, 'mix_both')  # TODO: 'mix_both' als subfolder inklusive for schleife
        for file in sorted(os.listdir(subdir_path)):
            # Path including .wav attachment
            file_path = os.path.join(subdir_path, file)
            # Read time series and scale to float32 Wertebereich
            time_series = len(wavfile.read(file_path)[1])
            number_of_entries = number_of_entries + time_series
            number_of_files = number_of_files + 1
        print('Gesamtstunden für \'{}\': '.format(single_dataset), (number_of_entries/sampling_rate/3600))
        print('Anzahl der Files für \'{}\': '.format(single_dataset), number_of_files)


if __name__ == '__main__':
    for i in ['min', 'max']:
        print(i)
        calculate_hours(i, sampling_rate=8000)
