# MA23


## Data Source
- **Original Data**: PhysioNet training A-F
- **Noise Data**: Some random files From DCASE like glass break, etc. (It is just for testing till using the right data)

## Data Preprocessing

- [ ] read and resample the original file and the noise file should be 44100 Hz,  1 Channcel , 16 bit

- [ ] Trimming: Check the duration of the original and the noise, if the noise duration more than 25% of the original file, Trim the noise file to be 25% of the original duration


- [ ] Amplify the original data with a factor calculated by noise_rms / orig_rms if the noise_rms is larger than orig_rms
- [ ] Normalize both orginal data and noise data


- [ ] add the noise to the original file, so that the noise can start in random position between the begin of the original file and maximum the 60% of the original file

- [ ] Generate the wavform, Frequency Domain and MFCC Plots and save them in a separate folder

- [ ] Log the prcocess in csv file which should have the following columns (original_filename, noise_filename, target_filename , original_duration, noise_duration, target duration, start_position of noise in seconds, end position of noise in seconds, original file_ sample rate, noise sample rate, target sample rate )





## Checklist 


## Sources
- ConvNetTas (original code: using MUSDB18): https://github.com/paxbun/Conv-TasNet/tree/master
- 



