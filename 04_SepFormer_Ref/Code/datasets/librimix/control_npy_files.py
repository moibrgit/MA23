import numpy as np
import os


def control_npy_files(general_path, float_dataset=False):
    """ Goes through each subfolder of a given path and checks if the created npy-files contain the correct values.
    :param general_path: Path from which the folders 'dev', 'test', 'train-100' and 'train-360' are visible
    :param float_dataset: normalized or unnormalized data set
    :return: Number of time series sections that not correctly decomposed
    """
    # Control for every dataset
    for i in ['dev', 'test', 'train-100', 'train-360']:
        mix_both = np.load(os.path.join(general_path, '{}/mix_both/librimix_as_array.npy'.format(i)))
        mix_clean = np.load(os.path.join(general_path, '{}/mix_clean/librimix_as_array.npy'.format(i)))
        noise = np.load(os.path.join(general_path, '{}/noise/librimix_as_array.npy'.format(i)))
        s1 = np.load(os.path.join(general_path, '{}/s1/librimix_as_array.npy'.format(i)))
        s2 = np.load(os.path.join(general_path, '{}/s2/librimix_as_array.npy'.format(i)))

        # Check for both possibilities
        if float_dataset:
            threshold = 3 / np.iinfo(np.int16).max
        else:
            threshold = 3

        # Mix-Both
        result_mix_both = mix_both - noise - s1 - s2
        count = np.count_nonzero(np.any(result_mix_both > threshold, axis=1))
        print('[{}] Number of wrong mix_both values: '.format(i), count)

        # Mix-Clean
        result_mix_clean = mix_clean - s1 - s2
        count = np.count_nonzero(np.any(result_mix_clean > threshold, axis=1))
        print('[{}] Number of wrong mix_clean values: '.format(i), count)


if __name__ == '__main__':
    control_npy_files('/isip/Public/spruenken/Libri2Mix/wav8k/min', float_dataset=False)
