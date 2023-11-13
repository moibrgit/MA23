import os
import tensorflow as tf
import numpy as np
import librosa
import pandas as pd
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sepformer import Sepformer  # Ensure this is correctly imported
from loss import SiSNR  
from sisnr import SiSNRLoss 
import random

os.chdir(Path(__file__).parent)

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(filename)s - %(lineno)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

# Constants
SEED = 50
random.seed(SEED)
tf.random.set_seed(SEED)

DATA_PATH = Path("../00_Dataset/Datenbank")
BASE_SAVE_PATH = Path("./models")
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
USE_GPU = False  # Set to False to use CPU, or True for GPU

# Create subfolder with datetime
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
SUBFOLDER_PATH = BASE_SAVE_PATH / f"Training_{current_time}"
try:
    SUBFOLDER_PATH.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created training subfolder: {SUBFOLDER_PATH}")
except Exception as e:
    logging.error(f"Error creating training subfolder: {e}")




# Load and preprocess audio data
def load_audio_data(file_path):
    file_names, audios, labels = [], [], []
    try:
        for file in file_path.glob("*.wav"):
            audio, _ = librosa.load(file, sr=None)
            audios.append(audio)
            label = file.stem.split('-')[0]
            labels.append(label)
            file_names.append(file.name)
        logging.info("Audio data loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading audio data: {e}")
    return file_names, audios, labels


def sisnr_loss(y_true, y_pred):
    # SiSNR Loss Function
    #     
    # return SiSNR(y_true, y_pred)
    # return SiSNRLoss()
    return SiSNRLoss(y_true, y_pred)


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(SUBFOLDER_PATH / 'training_progress.csv', 'a') as f:
            if epoch == 0:
                f.write(','.join(logs.keys()) + '\n')
            f.write(','.join(str(log) for log in logs.values()) + '\n')

# Train the model
def train_model(model, train_data, val_data):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(optimizer=optimizer, loss=SiSNRLoss())
    callbacks = [TrainingProgressCallback()]
    
    logging.info("Start model fitting")
    
    
    history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
    
    logging.info("Finished model fitting")
    
    return history

# Convert and preprocess audio data
def preprocess_data(audios, labels):
    # Assuming all audio files are of the same sample rate and length
    # Convert audios and labels to NumPy arrays
    audios_np = np.array(audios, dtype=object)
    labels_np = np.array(labels)
    return audios_np, labels_np

# Create TensorFlow Dataset for batching
def prepare_dataset(audios, labels, batch_size, dataset_name ):
    logging.debug(f"Preparing {dataset_name} started : {len(audios)} - {len(labels)}")

     
   

    dataset = tf.data.Dataset.from_tensor_slices((audios, labels))
    dataset = dataset.batch(batch_size)

    for batch in dataset.take(1):  # Take just the first batch of the dataset
        audios1, labels1 = batch
        print("Audio data type:", audios1.dtype)
        print("Labels data type:", labels1.dtype)

 

    return dataset

# Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    # Note: If 'accuracy' is not in history, replace it with the correct metric
    plt.plot(history.history.get('accuracy', []), label='Training Accuracy')
    plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.savefig(SUBFOLDER_PATH / 'training_history.png')
    plt.close()

if __name__ == "__main__":
    logging.info("Starting main execution")
    try:       

        if USE_GPU:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "2" # or "0"            
            logging.info("Using GPU for training")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            logging.info("Using CPU for training")

        file_names, audio_data, labels = load_audio_data(DATA_PATH)
        train_files, test_files, train_labels, test_labels = train_test_split(audio_data, labels, test_size=0.2)
        train_files, val_files, train_labels, val_labels = train_test_split(train_files, train_labels, test_size=0.1)

        logging.info("Preparing Training Data started")
        train_dataset = prepare_dataset(train_files, train_labels, BATCH_SIZE, "Train Dataset")
         

        logging.info("Preparing Validation Data started")
        val_dataset = prepare_dataset(val_files, val_labels, BATCH_SIZE, "Val Dataset")


       
        
        sepformer_model = Sepformer()
        logging.info("Sepformer model initialized")

        history = train_model(sepformer_model, train_dataset, val_dataset)
        logging.info("Training completed")

        sepformer_model.save(SUBFOLDER_PATH / 'model')
        sepformer_model.save_weights(SUBFOLDER_PATH / 'weights')
        logging.info("Model and weights saved")

        plot_history(history)
        logging.info("Training history plot saved")

    except Exception as e:
        logging.error(f"Error during main execution: {e}")
