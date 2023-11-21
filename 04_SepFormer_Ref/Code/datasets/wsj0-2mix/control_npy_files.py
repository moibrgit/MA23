import numpy as np
import os


def control_npy_files(general_path, float_dataset=False):
    """ Goes through each subfolder of a given path and checks if the created npy-files contain the correct values.
    :param float_dataset: normalized or unnormalized data set
    :param general_path: Path from which the folders 'dev', 'test', 'train-100' and 'train-360' are visible
    :return: Number of time series sections that not correctly decomposed
    """
    # Control for every dataset
    for i in ['dev', 'test', 'train']:
        mix_clean = np.load(os.path.join(general_path, '{}/mix_clean/wsj0-2mix_as_array.npy'.format(i)))
        s1 = np.load(os.path.join(general_path, '{}/s1/wsj0-2mix_as_array.npy'.format(i)))
        s2 = np.load(os.path.join(general_path, '{}/s2/wsj0-2mix_as_array.npy'.format(i)))

        # Mix-Clean
        result_mix_clean = mix_clean - s1 - s2
        if float_dataset:
            count = np.count_nonzero(np.any(result_mix_clean > 2 / np.iinfo(np.int16).max, axis=1))
        else:
            count = np.count_nonzero(np.any(result_mix_clean > 2, axis=1))
        print('[{}] Number of wrong mix_clean values: '.format(i), count)


if __name__ == '__main__':
    control_npy_files('/isip/Public/spruenken/wsj0-2mix_normed/wav8k/min', float_dataset=True)
