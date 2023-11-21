import yaml
import os
import shutil
import itertools
import numpy as np
import datetime

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from tinySepformer import TinySepformer
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Setup
path_to_src = '/isip/Users/spruenken/Documents/Thesis/src'
with open(os.path.join(path_to_src, "train_config.yaml"), "r") as f:
    config = yaml.safe_load(f)


def load_npy_tensors(dataset_type):
    """ Load converted audio files (Numpy arrays) as a training or validation dataset
    :param dataset_type: Type of data set ('train', 'val')
    :return: EagerTensor tuple (feature, labels)
    :rtype: tuple
    """
    
    if dataset_type == 'train':
        path_to_dataset = os.path.join(config['loadTensor']['pathLibrimixAsNpy'],
                                       config['loadTrainTensor']['trainType'])
    elif dataset_type == 'val':
        path_to_dataset = os.path.join(config['loadTensor']['pathLibrimixAsNpy'], 'dev')

    # Loading .npy files (train and two labels) of a specific configuration
    #feature_data = np.load(os.path.join(path_to_dataset, config['loadTensor']['mixType'], 'wsj0-2mix_as_array.npy'))
    #label1_data = np.load(os.path.join(path_to_dataset, 's1', 'wsj0-2mix_as_array.npy'))
    #label2_data = np.load(os.path.join(path_to_dataset, 's2', 'wsj0-2mix_as_array.npy'))
    feature_data = np.load(os.path.join(path_to_dataset, config['loadTensor']['mixType'], 'librimix_as_array.npy'))
    label1_data = np.load(os.path.join(path_to_dataset, 's1', 'librimix_as_array.npy'))
    label2_data = np.load(os.path.join(path_to_dataset, 's2', 'librimix_as_array.npy'))

    # Convert all three numpy-arrays into a tensor
    feature_data = tf.expand_dims(tf.convert_to_tensor(feature_data, dtype=tf.float64), -1)
    label1_data = tf.expand_dims(tf.convert_to_tensor(label1_data, dtype=tf.float64), -1)
    label2_data = tf.expand_dims(tf.convert_to_tensor(label2_data, dtype=tf.float64), -1)
    labels_data = tf.stack([label1_data, label2_data], axis=1)

    # Create a dataset tensor from given training data
    compiled_tensor = tf.data.Dataset.from_tensor_slices((feature_data, labels_data))

    # Delete redundant tensors and free up disk space
    del feature_data, label1_data, label2_data, labels_data

    # Optional: shuffle the values inside the tensors
    if dataset_type == 'train' and config['loadTrainTensor']['shuffle']:
        compiled_tensor = compiled_tensor.shuffle(buffer_size=len(compiled_tensor))

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
        si_snr_one_combination_log = 10 * tf.math.log(tf.abs(si_snr_one_combination) + eps) / tf.math.log(
            tf.cast(10, dtype=tf.float64))
        si_snr_list.append(tf.reduce_mean(si_snr_one_combination_log, axis=1))
        # Next combination (currently static)
        y_pred = tf.reverse(y_pred, axis=[1])
    # Determine the logarithm of the averaged value
    si_snr = tf.stack(si_snr_list, axis=0)
    loss = tf.reduce_max(si_snr, axis=0)

    return -1 * loss


class SaveLossCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        super(SaveLossCallback, self).__init__()
        self.train_losses = {}
        self.val_losses = {}

    def on_epoch_end(self, epoch, logs=None):
        self.train_losses[epoch+1] = logs['loss']
        self.val_losses[epoch+1] = logs['val_loss']

    def call(self):
        return self.train_losses, self.val_losses


class SaveWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path, loss_values, period=10):
        super(SaveWeightsCallback, self).__init__()
        self.save_path = save_path
        self.backup_path = os.path.join(self.save_path, 'Backups')
        self.period = period
        self.loss_values = loss_values

    def on_epoch_end(self, epoch, logs=None):
        # Parameters
        train_loss, val_loss = self.loss_values
        if epoch == (self.params['epochs'] - 1):
            # Check if the folder already exists for the model
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            # Save the finished model
            self.model.save_weights(os.path.join(self.save_path, "model.h5"))
            # Training-Loss
            with open(os.path.join(self.save_path, "train-loss.yaml"), 'w') as train_loss_file:
                yaml.dump(train_loss, train_loss_file)
            # Val-Loss
            with open(os.path.join(self.save_path, "val-loss.yaml"), 'w') as val_loss_file:
                yaml.dump(val_loss, val_loss_file)
        elif (epoch + 1) % self.period == 0:
            # Create the directory path if it does not exist
            if not os.path.exists(self.backup_path):
                os.makedirs(self.backup_path)
            # Save the weights of the model and the loss values for the current epoch.
            weights_path = os.path.join(self.backup_path, f"epoch_{str(epoch+1)}.h5")
            self.model.save_weights(weights_path)
            # Training-Loss
            with open(os.path.join(self.backup_path, f"train-loss_epoch_{str(epoch+1)}.yaml"), 'w') as loss_file_epoch:
                yaml.dump(train_loss, loss_file_epoch)
            # Val-Loss
            with open(os.path.join(self.backup_path, f"val-loss_epoch_{str(epoch+1)}.yaml"), 'w') as loss_file_epoch:
                yaml.dump(val_loss, loss_file_epoch)


if __name__ == '__main__':
    # Parameters and initialization
    epochs = config['train']['epochs']
    batch_size = config['train']['batchSize']
    model_path = config['train']['modelPath']

    # ID for saving the model
    model_id = datetime.datetime.now()
    model_id = model_id.strftime("%Y-%m-%d_%H-%M-%S")

    # Save configs
    finish_path = os.path.join(model_path, model_id)
    if not os.path.exists(finish_path):
        os.makedirs(finish_path)
    shutil.copy(os.path.join(path_to_src, 'tinySepformer_config.yaml'),
                os.path.join(finish_path, 'tinySepformer_config.yaml'))
    shutil.copy(os.path.join(path_to_src, 'train_config.yaml'), os.path.join(finish_path, 'train_config.yaml'))

    # Convert npy-files into training and validation dataset
    with tf.device("CPU"):
        train_dataset = load_npy_tensors('train')
        train_dataset = train_dataset.batch(batch_size)
        val_dataset = load_npy_tensors('val')
        val_dataset = val_dataset.batch(batch_size)

    # Compile the model
    model = TinySepformer()
    optimizer = tf.keras.optimizers.Adam(learning_rate=config['train']['learningRate'], global_clipnorm=5)
    model.compile(optimizer=optimizer, loss=si_snr_loss)

    # Train and validate the model
    save_loss_callback = SaveLossCallback()
    save_weights_callback = SaveWeightsCallback(save_path=finish_path, loss_values=save_loss_callback.call(),  period=2)
    adaptive_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0)
    callbacks = [save_loss_callback, save_weights_callback, adaptive_learning_rate]

    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks)