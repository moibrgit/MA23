import numpy as np
import yaml
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
from tinySepformer import TinySepformer


def load_npy_test_data(data_set):
    path_to_test_data_sets = os.path.join('/isip/Public/spruenken', data_set, '0.25_without_zeros/min/dev')
    feature_npy_path = os.path.join(path_to_test_data_sets, 'mix_clean', '{}_as_array.npy'.format(data_set.lower()))
    s1_npy_path = os.path.join(path_to_test_data_sets, 's1', '{}_as_array.npy'.format(data_set.lower()))
    s2_npy_path = os.path.join(path_to_test_data_sets, 's2', '{}_as_array.npy'.format(data_set.lower()))

    # Load npy files (numpy arrays)
    feature_data = np.load(feature_npy_path)
    label1_data = np.load(s1_npy_path)
    label2_data = np.load(s2_npy_path)

    # Convert all three numpy-arrays into a tensor
    feature_data = tf.expand_dims(tf.convert_to_tensor(feature_data, dtype=tf.float64), -1)
    label1_data = tf.expand_dims(tf.convert_to_tensor(label1_data, dtype=tf.float64), -1)
    label2_data = tf.expand_dims(tf.convert_to_tensor(label2_data, dtype=tf.float64), -1)
    labels_data = tf.stack([label1_data, label2_data], axis=1)

    # Create a dataset tensor from given training data
    compiled_tensor = tf.data.Dataset.from_tensor_slices((feature_data, labels_data))

    # Delete redundant tensors and free up disk space
    del feature_data, label1_data, label2_data, labels_data

    return compiled_tensor


def si_snr_loss(y_true, y_pred):
    """Permutation invariant calculation of the loss function (for two speakers)
    :param y_true: Labels of all speaker
    :type y_true: tensor (Batch x Speaker x Time series x 1)
    :param y_pred: Predictions of all speaker
    :type y_pred: tensor (Batch x Speaker x Time series x 1)
    :return: Permutations invariant loss function
    :rtype: (EagerTensor) float64
    """

    # Number of speakers and possible combinations
    speakers = y_pred.shape[1]
    # combinations = list(itertools.combinations(range(speakers), 2))

    # Subtracting the mean value
    #y_pred = y_pred - tf.reduce_mean(y_pred, axis=2, keepdims=True)

    # Calculate the SI-SNR for each combination (still static for two speakers)
    eps = 1e-15
    si_snr_list = []
    for i in range(2):
        # s_target
        s_norm_sqared = tf.expand_dims(tf.square(tf.norm(y_true, ord=2, axis=2) + eps), axis=-1)
        sroof_s_dotproduct = tf.expand_dims(tf.reduce_sum(y_pred * y_true, axis=2), -1)
        s_target = tf.cast(tf.divide(tf.multiply(sroof_s_dotproduct, y_true), s_norm_sqared), dtype=tf.float64)
        # e_noise
        e_noise = y_pred - s_target
        # Average for each batch
        si_snr_one_combination = tf.divide(tf.square(tf.norm(s_target, ord=2, axis=2)),
                                           tf.square(tf.norm(e_noise, ord=2, axis=2) + eps))
        si_snr_one_combination_log = 10 * tf.math.log(tf.abs(si_snr_one_combination)) / tf.math.log(
            tf.cast(10, dtype=tf.float64))
        si_snr_list.append(tf.reduce_mean(si_snr_one_combination_log, axis=1))
        # Next combination (currently static)
        y_pred = tf.reverse(y_pred, axis=[1])
    # Determine the logarithm of the averaged value
    si_snr = tf.stack(si_snr_list, axis=0)
    loss = tf.reduce_max(si_snr, axis=0)

    return -1 * loss


if __name__ == '__main__':

    # Parameters
    model = TinySepformer()
    all_models = '/isip/Users/spruenken/Documents/Models'
    model_weights = '2023-07-22_12-35-21'
    data_set = 'LibriMix'  # 'LibriMix' -- 'wsj0-2mix'
    intermediate_path = True
    epoch = 42  # only necessary if intermediate status is to be tested
    batch_size = 8

    model.compile(loss=si_snr_loss)

    # Convert npy-files into training and validation dataset
    with tf.device("CPU"):
        test_dataset = load_npy_test_data(data_set=data_set)
        test_dataset = test_dataset.batch(batch_size)

    # Test model in 'Backups' path
    if intermediate_path:
        path_to_models = os.path.join(all_models, model_weights, 'Backups')
        model_file = 'epoch_{}.h5'.format(epoch)
        entire_path = os.path.join(path_to_models, model_file)
    # Test completed model
    else:
        path_to_models = os.path.join(all_models, model_weights)
        model_file = 'model.h5'
        entire_path = os.path.join(path_to_models, model_file)

    # Load the model
    output = model(tf.ones([1, 2000, 1]))
    model.load_weights(entire_path, by_name=True)

    # Test the model
    model.evaluate(test_dataset)
