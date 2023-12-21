# Run: /isip/Public/pallenberg/venv/sepformerTaylorTFPytorch/bin/python3.10 

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
from loss import SiSNR, SDR  
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

# DATA_PATH_MIXED = Path("../00_Dataset/MyMerged/20231116_110630/Data/")
# DATA_PATH_CLEAN = Path("../00_Dataset/DS_PhysioNet/training/all/")

DATA_PATH_MIXED = Path("./Data/mixed/")
DATA_PATH_CLEAN = Path("./Data/clean")

DATA_PATH_MIXED_TRAIN = DATA_PATH_MIXED / "train"
DATA_PATH_MIXED_VAL= DATA_PATH_MIXED / "val"


TENSOR_SHUFFLE = True

BASE_SAVE_PATH = Path("./models")
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 0.001
SR = 8000 # or None

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
    file_names, audios_mixed, audios_clean = [], [], [] 
    try:
        for file_mixed in file_path.rglob("*.wav"):

            # Load mixed audio file
            audio_mixed, _ = librosa.load(file_mixed, sr=SR)
            audios_mixed.append(audio_mixed)
            
            # get the clean file name from the audio_mixed_file name --> merg_a0001_173559.wav -> a0001.wav
            
            clean_file_name = file_mixed.stem.split('_')[1] + '.wav'
            
            clean_file_path = DATA_PATH_CLEAN / clean_file_name

            # Load clean audio file --> My Label
            if clean_file_path.exists():            
                audio_clean, _ = librosa.load(clean_file_path, sr=SR)
            else:
                logging.error("Clean Audio: {clean_file_name} is not found")        

            print(f"Clean: {audio_clean.shape}")
            print(f"Mixed: {audio_mixed.shape}")

            audio_mixed = audio_mixed[:audio_clean.shape[0]]

            audios_clean.append(audio_clean)
            file_names.append(file_mixed.name)

            print(f"Clean: {audio_clean.shape}")
            print(f"Mixed: {audio_mixed.shape}")

        logging.debug("Audio data loaded successfully. {file.name}")
    except Exception as e:
        logging.error(f"Error loading audio data : {e}")
    return file_names, audios_mixed, audios_clean


def sisnr_loss():
    # SiSNR Loss Function
    #     
    # return SiSNR(y_true, y_pred)
    
    return SiSNR()
    return SDR()
    
    # return SiSNRLoss()
    return SiSNRLoss(y_true, y_pred)


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        with open(SUBFOLDER_PATH / 'training_progress.csv', 'a') as f:
            if epoch == 0:
                f.write(','.join(logs.keys()) + '\n')
            f.write(','.join(str(log) for log in logs.values()) + '\n')


def train_model(model, train_data, val_data):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    # model.compile(optimizer= optimizer, loss=sisnr_loss())
    # model.compile(optimizer= optimizer, loss=sisnr_loss)
    model.compile(optimizer= optimizer, loss=SiSNR())
    
    # Callbacks
    # adaptive_learning_rate = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0)
    callbacks = [TrainingProgressCallback()]    
    # callbacks = [TrainingProgressCallback(), adaptive_learning_rate]    
    
    
    logging.info("Start model fitting")
    
    
    # Fixme, valudation data should be separate
    history = model.fit(train_data, validation_data = val_data, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)
    
    logging.info("Finished model fitting")
    
    return history


    # return dataset

def prepare_dataset(audios_mixed, audios_clean, batch_size):
  
  
    logging.debug(f"Preparing started : {len(audios_mixed)} - {len(audios_clean)}")

    # Convert Numpy Arrays to tensor and extend the dim
    audios_mixed = tf.expand_dims(tf.convert_to_tensor(audios_mixed, dtype=tf.float64), -1)
 
    # Create a dataset tensor from given training data
    dataset_tensor = tf.data.Dataset.from_tensor_slices((audios_mixed, audios_clean))

    if TENSOR_SHUFFLE:
        dataset_tensor = dataset_tensor.shuffle(buffer_size=len(dataset_tensor))
    
    dataset_tensor = dataset_tensor.batch(batch_size)

    return dataset_tensor

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
            os.environ["CUDA_VISIBLE_DEVICES"] = "2" # or "0"  or "1"
            
            
            import script_gpu_torch_based
            script_gpu_torch_based.mask_unused_gpus(needed_memory=5000)
            script_gpu_torch_based.tf_set_memory_usage_dynamic()

            
            # gpus = tf.config.experimental.list_physical_devices('GPU')
            # for gpu in gpus:
            #     tf.config.experimental.set_memory_growth(gpu, True)
                        
            logging.info("Using GPU for training")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            logging.info("Using CPU for training")



         
        
        logging.info("Loading Data started")
        file_names_mixed_train, audios_mixed_train, audios_clean_train = load_audio_data(DATA_PATH_MIXED_TRAIN)
        file_names_mixed_val, audios_mixed_val, audios_clean_val = load_audio_data(DATA_PATH_MIXED_VAL)

        # Split data into train and test
        # train_audios_mixed, test_audios_mixed, train_audios_clean, test_audios_clean = train_test_split(audios_mixed, audios_clean, test_size=0.2)
       

        logging.info("Preparing Data started")
        train_dataset = prepare_dataset(audios_mixed_train, audios_clean_train, BATCH_SIZE)
        val_dataset = prepare_dataset(audios_mixed_val, audios_clean_val, BATCH_SIZE)
         
      
        
        # Initialize the model
        logging.info("Sepformer model initialized")
        sepformer_model = Sepformer()
        

        # Training the model
        logging.info("Training started")
        history = train_model(sepformer_model, train_dataset, val_dataset )
        logging.info("Training completed")


        # Saving the weights and model
        sepformer_model.save(SUBFOLDER_PATH / 'model')
        sepformer_model.save_weights(SUBFOLDER_PATH / 'weights')
        logging.info("Model and weights saved")

        plot_history(history)
        logging.info("Training history plot saved")

    except Exception as e:
        logging.error(f"Error during main execution: {e}")
