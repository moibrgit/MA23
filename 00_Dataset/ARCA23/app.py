__author__ = "Mohamed Ibrahim "

import os 
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging
import soundfile as sf



os.chdir(Path(__file__).parent)


# Initialize logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])


# Function to create a directory
def create_directory(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


# Function to analyze audio file
def analyze_audio(file_path):
    with sf.SoundFile(file_path) as audio:
        bit_depth = audio.subtype.replace('PCM_', '')
        channels = audio.channels
        sample_rate = audio.samplerate
    return bit_depth, channels, sample_rate

# Function to process each CSV file
def process_csv(csv_file, categories, source_folder, output_folder):
    df = pd.read_csv(csv_file)
    filtered_df = df[df['label'].isin(categories)]

    # Appending ".wav" to each filename
    filtered_df['fname'] = filtered_df['fname'].apply(lambda x: f"{x}.wav")

    target_folder = Path(output_folder) / csv_file.split('.')[0]
    target_folder.mkdir(parents=True, exist_ok=True)

    audio_info = []
    for _, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0]):
        src_file = Path(source_folder) / row['fname']
        dest_file = target_folder / row['fname']


        if src_file.exists():
            shutil.copy(src_file, dest_file)
            bit_depth, channels, sample_rate = analyze_audio(dest_file)
            audio_info.append([row['fname'], row['label'], bit_depth, channels, sample_rate])

            new_df = pd.DataFrame(audio_info, columns=['filename', 'label', 'bit_depth', 'channels', 'sample_rate'])
            new_df.to_csv(target_folder / f'{csv_file.split(".")[0]}_info.csv', index=False)
            logging.info(f'Processed {csv_file} and copied files to {target_folder}')

# Main function
def main():
    output_folder = create_directory('./output/' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    categories = ["Cough", "Child_speech_and_kid_speaking", "Wind"]
    # categories = ["Wind", "Wind_chime"]

    csv_files = ['./ARCA23K-FSD.ground_truth/train.csv', 
                 './ARCA23K-FSD.ground_truth/test.csv', 
                 './ARCA23K-FSD.ground_truth/val.csv']
    source_folder = './ARCA23K.audio'

    for csv_file in csv_files:
        process_csv(csv_file, categories, source_folder, Path(output_folder))

if __name__ == "__main__":
    main()
