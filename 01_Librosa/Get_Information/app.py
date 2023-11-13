import os
import librosa
import csv
import numpy as np
import logging 
from pathlib import Path

os.chdir(Path(__file__).parent)

# Initialize Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

def get_audio_info(audio_path):
    """
    Extracts important information from an audio file.
    """

    

    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_centroid_mean = np.mean(spectral_centroids)
    spectral_centroid_std = np.std(spectral_centroids)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_std = np.std(spectral_bandwidth)

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Zero-Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zero_crossing_rate)
    zcr_std = np.std(zero_crossing_rate)

    return {
        'filename': audio_path.name,
        'sample_rate': sr,
        'duration_seconds': duration,
        'spectral_centroid_mean': spectral_centroid_mean,
        'spectral_centroid_std': spectral_centroid_std,
        'spectral_bandwidth_mean': spectral_bandwidth_mean,
        'spectral_bandwidth_std': spectral_bandwidth_std,
        'tempo': tempo,
        'zero_crossing_rate_mean': zcr_mean,
        'zero_crossing_rate_std': zcr_std
    }

def process_folder(folder_path, output_csv):
    """
    Processes all audio files in the given folder and writes the information to a CSV file.
    """
    
    audio_files = Path(folder_path).glob('*.wav')  # Update the extension if needed

    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = [
            'filename', 'sample_rate', 'duration_seconds',
            'spectral_centroid_mean', 'spectral_centroid_std',
            'spectral_bandwidth_mean', 'spectral_bandwidth_std',
            'tempo', 'zero_crossing_rate_mean', 'zero_crossing_rate_std'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for audio_file in audio_files:
            logging.info(f"getting audio information : {Path(audio_file).name} ")
            info = get_audio_info(audio_file)
            writer.writerow(info)
            logging.info(f"Saved information about : {Path(audio_file).name} ")
            

def main():
    
    folder_path = '../../00_Dataset/Datenbank'  # Update with the actual folder path
    output_csv = './audio_info.csv'

    logging.info("*** Application Started *** ")

    process_folder(folder_path, output_csv)

    logging.info("*** Application finished *** ")

if __name__ == "__main__":
    main()
