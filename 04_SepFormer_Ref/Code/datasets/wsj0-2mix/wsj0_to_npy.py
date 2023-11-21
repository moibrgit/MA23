import os
import numpy as np
from scipy.io import wavfile

# Path to the LibriMix folder and selected subfolder
wsj0_path = '/isip/Students-ro/SpeechSeparation/wsj0-mix'
public_path = '/isip/Public/spruenken/wsj0-2mix_normed/'
subfolder = ['mix', 's1', 's2']


def bundle_wav_files(dataset, cutout_size=1, wsj0_mix='wsj0-2mix', sampling_rate=8000, sampling_length='min'):
    """ Convert the .wav-files of a given data set into an 2D-array (.npy-file)
    :param dataset: 'cv', 'tr', 'tt'
    :param cutout_size: length of cutout in seconds
    :param wsj0_mix: 'wsj0-2mix', 'wsj0-3mix' (static für 2mix)
    :param sampling_rate: '8000', '16000'
    :param sampling_length: 'min', 'max'
    """
    # Paths


    # Calculate cutout size and select wav_type-variable
    cutout = sampling_rate * cutout_size
    wav_type = "wav{}k".format(sampling_rate // 1000)
    wsj0_mix = "2speakers"  # Static due to the given folder structure

    # Path to the subfolder containing the different types of data for the network
    folder_path = os.path.join(wsj0_path, wsj0_mix, wav_type, sampling_length, dataset)

    for subdir in subfolder:
        npy_array = []
        # Subfolder with type of sounds
        subdir_path = os.path.join(folder_path, subdir)
        for file in sorted(os.listdir(subdir_path)):
            # Path including .wav attachment
            file_path = os.path.join(subdir_path, file)
            # Read time series and scale to float32 Wertebereich
            time_series = wavfile.read(file_path)[1]
            time_series = time_series.astype(np.float32) / np.iinfo(np.int16).max
            # Split time series into one seconds sub-arrays
            n = len(time_series)
            for i in range(0, n-(n % cutout), cutout):
                sub_array = []
                sub_array = time_series[i:i+cutout]
                npy_array.append(sub_array)
        # Save the numpy array under the corresponding path
        arr = np.array(npy_array)
        # Adapt the storage path to that of LibriMix
        dataset_correlation = {'cv': 'dev', 'tr': 'train', 'tt': 'test'}
        mix_correlation = {'mix': 'mix_clean', 's1': 's1', 's2': 's2'}
        save_path = os.path.join(public_path, wav_type, sampling_length, dataset_correlation[dataset], mix_correlation[subdir])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = 'wsj0-2mix_as_array'
        np.save(os.path.join(save_path, save_name + '.npy'), arr)
        np.savez_compressed(os.path.join(save_path, save_name + '.npz'), arr=arr, allow_pickle=True, pickle_protocol=2)
        print('Saved under: ', os.path.join(save_path, save_name + '.npy'))


if __name__ == '__main__':
    for j in ['cv', 'tr', 'tt']:  # cv := val, tr := train, tt := test
        bundle_wav_files(j)

    print('Datensatz für 8000 wurde erstellt')

    for j in ['cv', 'tr', 'tt']:  # cv := val, tr := train, tt := test
        bundle_wav_files(j, sampling_rate=16000)
