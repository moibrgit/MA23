import os
import librosa
import soundfile as sf
import csv
import numpy as np
import logging , random
from pathlib import Path
from tqdm import tqdm 

""" 
- Sample Rate 
- Duration and Timing
- Volume Level & Normalization
- 

"""


os.chdir(Path(__file__).parent)

# Initialize Logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])



def calculate_rms(audio_data):
    """To calculate the RMS in order to determine the Amplification factor of the original file

    Args:
        audio_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    return np.sqrt(np.mean(np.square(audio_data)))


def match_sample_rate_and_duration(original_file, noise_file, target_sr, target_length, output_folder, csv_writer):
    # Load original and noise files
    orig_audio, orig_sr = librosa.load(original_file, sr=target_sr)
    noise_audio, noise_sr = librosa.load(noise_file, sr=target_sr)

    # Calculate durations in seconds
    orig_duration = librosa.get_duration(y=orig_audio, sr=target_sr)
    noise_duration = librosa.get_duration(y=noise_audio, sr=target_sr)

    # Trim noise if needed
    if noise_duration >= orig_duration:
        max_noise_length = orig_duration / 4 # Noise file should be max 1/4 of the orignal file 
        noise_audio = noise_audio[:int(max_noise_length * target_sr)]

    # Randomly position noise within the original audio
    max_start = len(orig_audio) - len(noise_audio)
    start_pos = random.randint(0, max_start)
    end_pos = start_pos + len(noise_audio)

    
    # Calculate RMS amplitude and adjust amplification
    orig_rms = calculate_rms(orig_audio)
    noise_rms = calculate_rms(noise_audio)

    if noise_rms > orig_rms:
        amplification_factor = noise_rms / orig_rms
        orig_audio = orig_audio * amplification_factor

    # Normalize volume after amplification
    orig_audio = librosa.util.normalize(orig_audio)
    noise_audio = librosa.util.normalize(noise_audio)


    # Overlap the orig_audio with noise_audio
    orig_audio[start_pos:end_pos] += noise_audio  # Insert noise into the original audio
    overlapped_audio = orig_audio
    
    # overlapped_audio = orig_audio + noise_audio

    
    


    # Trim or extend the target audio to the specified target length (5 seconds)
    target_samples = int(target_length * target_sr)
    if len(overlapped_audio) > target_samples:
        overlapped_audio = overlapped_audio[:target_samples]
    else:
        overlapped_audio = np.pad(overlapped_audio, (0, max(0, target_samples - len(overlapped_audio))), mode='constant')

    # Save the overlapped audio
    overlapped_file = output_folder / original_file.name
    sf.write(overlapped_file, overlapped_audio, target_sr)


    # Log information to CSV
    csv_writer.writerow({
        'original_file': original_file.name,
        'original_length': orig_duration,
        'noise_file': noise_file.name,
        'noise_length': noise_duration,
        'target_file': overlapped_file.name,
        'target_length': target_length,
        'start_position': 0,  # Assuming noise starts at the beginning
        'end_position': noise_duration,
        'target_sample_rate': target_sr
    })

def preprocess_and_overlap(original_folder, noise_folder, output_folder, target_sr,target_length, log_file):
    original_folder = Path(original_folder)
    noise_folder = Path(noise_folder)
    output_folder = Path(output_folder)

    original_files = list(original_folder.rglob('*.wav'))
    noise_files = list(noise_folder.rglob('*.wav'))
    total_orig_files = len(original_files)

    # random.shuffle(noise_files)

    with open(log_file, mode='w', newline='') as file:
        fieldnames = ['original_file', 'original_length', 'noise_file', 'noise_length', 
                      'target_file', 'target_length', 'start_position', 'end_position', 'target_sample_rate']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for orig_file in tqdm(original_files, total=total_orig_files, desc="Processing audio files"):
        # for orig_file in original_files:
            # Pair with a random noise file
            noise_file = random.choice(noise_files)
            match_sample_rate_and_duration(orig_file, noise_file, target_sr, target_length, output_folder, writer)


def main():
    original_folder = '../../00_Dataset/DS_PhysioNet/training'
    noise_folder = '../Audio_Overlapper/noise'
    output_folder = '../../00_Dataset/MyMerged/t1'
    log_file = '../../00_Dataset/MyMerged/log.csv'
    target_sr = 44100  # Target sample rate (in Hz), adjust as needed
    target_length = 5  # Target length in seconds
    
    preprocess_and_overlap(original_folder, noise_folder, output_folder, target_sr,target_length, log_file) 


if __name__ == "__main__":
    main()