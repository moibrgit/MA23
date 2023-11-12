__author__ = "Mohamed Ibrahim"

import csv
import datetime
import logging
import os
import random
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import soundfile as sf

os.chdir(Path(__file__).parent)

# Initialize Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

def merge_audio_files(normal_folder, noise_folder, output_base_folder, csv_file_name, timestamp):
    # Create a unique subfolder for this execution
    
    output_folder = output_base_folder / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)

    csv_data = []
    logging.info("Starting to process and merge audio files...")

    for normal_file in normal_folder.iterdir():
        if normal_file.is_file():
            noise_file = random.choice(list(noise_folder.glob('*')))
            
            # Load the audio files
            normal_audio, sr = librosa.load(str(normal_file), sr=None)
            noise_audio, _ = librosa.load(str(noise_file), sr=sr)

            # Randomly select start position for the noise (Sample position and not time)
            start_pos = random.randint(0, len(normal_audio) - len(noise_audio))
            end_pos = start_pos + len(noise_audio)

            # Convert start position to time in seconds
            start_pos_seconds = start_pos / sr
            end_pos_seconds = end_pos / sr

            # Add the noise to the normal audio
            merged_audio = normal_audio.copy()
            merged_audio[start_pos:end_pos] += noise_audio

            # Save the merged audio
            output_file_name = f'merged_{normal_file.name}'
            output_path = output_folder / output_file_name
            sf.write(str(output_path), merged_audio, sr)
            logging.info(f"Merged file saved: {output_file_name}")

            # Append information to CSV data
            csv_data.append([normal_file.name, noise_file.name, output_file_name, start_pos_seconds, end_pos_seconds])

    # Write data to CSV
    csv_path = output_folder / csv_file_name
    with csv_path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Normal File', 'Noise File', 'Output File', 'Start Noise Position', 'End Noise Position'])
        writer.writerows(csv_data)
    logging.info("CSV file with merge information saved.")

def visualize_and_save_features(merged_folder, imgs_folder, timestamp):
    
    logging.info("Starting to visualize and save audio features...")

    # Create a unique subfolder for this execution
    output_folder = imgs_folder / timestamp
    output_folder.mkdir(parents=True, exist_ok=True)

    audio_file_list = list(merged_folder.glob("*.wav"))

    # for audio_file in merged_folder.iterdir():
    for audio_file in audio_file_list:
        if audio_file.is_file():
            audio, sr = librosa.load(str(audio_file), sr=None)

            # Compute MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr)

            # Plotting
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(mfccs, x_axis='time')
            plt.colorbar()
            plt.title(f'MFCC of {audio_file.name}')
            plt.tight_layout()

            # Save the plot
            image_file_name = f'mfcc_{audio_file.name}.png'
            image_path = output_folder / image_file_name
            plt.savefig(str(image_path))
            plt.close()
            logging.info(f"Saved visualization: {image_file_name}")



def main():
    normal_folder = Path('./normal')
    noise_folder = Path('./noise')
    output_base_folder = Path('merged')
    csv_file = './audio_merge_info.csv'
    imgs_folder = Path('./imgs')


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.info(f"*** Application Started with Folder Name *** : {timestamp}")

    # Process and merge audio files
    merge_audio_files(normal_folder, noise_folder, output_base_folder, csv_file, timestamp)

    # Visualize and save features
    
    visualize_and_save_features(output_base_folder / timestamp, imgs_folder, timestamp)

    logging.info(f"*** Application Finished Successfully *** : {timestamp}")


if __name__  == "__main__":
    main()