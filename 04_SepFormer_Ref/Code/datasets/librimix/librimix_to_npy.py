import os
import numpy as np
from scipy.io import wavfile

# Path to the LibriMix folder and selected subfolder
librimix_path = '/isip/Students-ro/SpeechSeparation/LibriMix/LibriMix'
public_path = '/isip/Public/spruenken/Libri2Mix/'
subfolder = ['mix_both', 'mix_clean', 'mix_single', 'noise', 's1', 's2']


def bundle_wav_files(dataset, cutout_size=1, librimix='Libri2Mix', sampling_rate=8000, sampling_length='min'):
    """ Convert the .wav-files of a given data set into an 2D-array (.npy-file)
    :param dataset: 'dev', 'test', 'train-100', 'train-360'
    :param cutout_size: length of cutout in seconds
    :param librimix: 'Libri2Mix', 'Libri3Mix
    :param sampling_rate: '8000', '16000'
    :param sampling_length: 'min', 'max'
    """
    # Calculate cutout size and select wav_type-variable
    cutout = int(sampling_rate * cutout_size)
    wav_type = "wav{}k".format(sampling_rate // 1000)

    # Path to the subfolder containing the different types of data for the network
    folder_path = os.path.join(librimix_path, librimix, wav_type, sampling_length, dataset)

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
        save_path = os.path.join(public_path, str(cutout_size), sampling_length, dataset, subdir)  # TODO: wav_type statt cutout_size
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = 'librimix_as_array'
        np.save(os.path.join(save_path, save_name + '.npy'), arr)
        np.savez_compressed(os.path.join(save_path, save_name + '.npz'), arr=arr, allow_pickle=True, pickle_protocol=2)


if __name__ == '__main__':
    for j in ['dev', 'test', 'train-100', 'train-360']:
        bundle_wav_files(j, cutout_size=1)