import os
import librosa
import numpy as np
import pandas as pd
import random
import logging
import csv
from pathlib import Path
import soundfile as sf
from tqdm import tqdm 
import datetime

import matplotlib.pyplot as plt
import librosa.display

# Constants

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# NOW_FOLDER = "data"
NOW_FOLDER = timestamp

ORIGINAL_FOLDER = Path('../../00_Dataset/DS_PhysioNet/training/')
NOISE_FOLDER = Path('../../01_Librosa/Audio_Overlapper/noise/')

MERGE_FOLDER = Path(f'../../00_Dataset/MyMerged/{NOW_FOLDER}')
DATA_FOLDER = MERGE_FOLDER / "Data"
IMAGES_FOLDER = MERGE_FOLDER / "Img"

LOG_FILE = MERGE_FOLDER / "process_log.csv"


SAMPLE_RATE = 44100
MAX_NOISE_START_PERCENTAGE = 0.60
MAX_NOISE_DURATION_PERCENTAGE = 0.25

# Set up logging
os.chdir(Path(__file__).parent)

# Make Outputfolder
MERGE_FOLDER.mkdir(parents=True, exist_ok=True)
DATA_FOLDER.mkdir(parents=True, exist_ok=True)
IMAGES_FOLDER.mkdir(parents=True, exist_ok=True)



# Initialize Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("./app2.log"),
                        # logging.StreamHandler()
                    ])


csv_header = ['original_filename', 'noise_filename', 'target_filename', 'original_duration', 
              'noise_duration', 'target_duration', 'start_position', 'end_position', 
              'original_sample_rate', 'noise_sample_rate', 'target_sample_rate']
with open(LOG_FILE, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

# Function to read and process wav file
def read_wav(file_path):
    data, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return sr, data

def calculate_rms(audio_data):
    """To calculate the RMS in order to determine the Amplification factor of the original file

    Args:
        audio_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    return np.sqrt(np.mean(np.square(audio_data)))


# Function to add noise
def add_noise(original, noise, start_pos):
    end_pos = start_pos + len(noise)
    if end_pos > len(original):
        end_pos = len(original)
        noise = noise[:end_pos - start_pos]
    original[start_pos:end_pos] += noise
    return original


def generate_plots(audio_data, sr, start_pos, end_pos, file_name):
    # Time in seconds for the x-axis -> to align all plots under each other
    time = np.linspace(0, len(audio_data) / sr, num=len(audio_data))

    plt.figure(figsize=(10, 9))

    # Time-domain plot
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(audio_data, sr=sr)
    plt.scatter([start_pos/sr, end_pos/sr], [0, 0], color='red')  # Scatter for start and end
    plt.title('Time Domain Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.xlim(0, time[-1])  # Set x-axis limits to match time range -> to remove the inside padding of the plot

    # Frequency-domain plot
    plt.subplot(3, 1, 2)
    D = np.abs(librosa.stft(audio_data))
    db = librosa.amplitude_to_db(D, ref=np.max)
    librosa.display.specshow(db, sr=sr, x_axis='time', y_axis='log')
    plt.scatter([start_pos/sr, end_pos/sr], [db.min(), db.min()], color='red')  # Scatter for start and end
    plt.title('Frequency Domain Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')

    # MFCC plot
    plt.subplot(3, 1, 3)    
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.scatter([start_pos/sr, end_pos/sr], [0, 0], color='red')  # Scatter for start and end
    plt.title('MFCC')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC')

    # Save the plot
    image_path = IMAGES_FOLDER / f'{file_name}.png'
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()


# Processing function
def process_files():

    original_files = list(ORIGINAL_FOLDER.rglob('*.wav'))
    noise_files = list(NOISE_FOLDER.rglob('*.wav'))
    total_orig_files = len(original_files)

    # total_noise_files = len(noise_files)


    for original_file in tqdm(original_files, total=total_orig_files, desc="Original files"):
        for noise_file in noise_files:
            original_sr, original_data = read_wav(original_file)
            noise_sr, noise_data = read_wav(noise_file)

            # Trim noise if necessary
            max_noise_length = int(len(original_data) * MAX_NOISE_DURATION_PERCENTAGE)
            if len(noise_data) > max_noise_length:
                noise_data = noise_data[:max_noise_length]

             # Calculate RMS amplitude and adjust amplification
            orig_rms = calculate_rms(original_data)
            noise_rms = calculate_rms(noise_data)

            if noise_rms > orig_rms:
                amplification_factor = noise_rms / orig_rms
                # orig_audio = original_data * amplification_factor
                original_data = original_data * amplification_factor

            # Normalize volume after amplification
            original_data = librosa.util.normalize(original_data)
            noise_data = librosa.util.normalize(noise_data)

            # Determine random start position for noise
            max_start_pos = int(len(original_data) * MAX_NOISE_START_PERCENTAGE)
            start_pos = random.randint(0, max_start_pos)

            # Add noise to original
            processed_data = add_noise(original_data, noise_data, start_pos)

            # Save the new file
            target_file = DATA_FOLDER / f'merg_{original_file.stem}_{noise_file.stem}.wav'
            sf.write(target_file, processed_data, SAMPLE_RATE)

            # Generate and save the plots
            end_pos = start_pos + len(noise_data)
            generate_plots(processed_data, SAMPLE_RATE, start_pos, end_pos, target_file.stem)



            # Log the process
            log_data = [original_file.name, noise_file.name, target_file.name, 
                        len(original_data) / original_sr, len(noise_data) / noise_sr, 
                        len(processed_data) / SAMPLE_RATE, start_pos / SAMPLE_RATE, 
                        (start_pos + len(noise_data)) / SAMPLE_RATE, original_sr, noise_sr, SAMPLE_RATE]
            with open(LOG_FILE, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(log_data)

           

            logging.info(f"Processed {original_file.name} with {noise_file.name}")

# Run the processing function
process_files()
