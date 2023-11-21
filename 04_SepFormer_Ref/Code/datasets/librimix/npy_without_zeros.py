import numpy as np
import os

# Einladen der npy-Dateien
mix_clean = np.load('/isip/Public/spruenken/LibriMix/0.25/min/test/mix_clean/librimix_as_array.npy')
s1 = np.load('/isip/Public/spruenken/LibriMix/0.25/min/test/s1/librimix_as_array.npy')
s2 = np.load('/isip/Public/spruenken/LibriMix/0.25/min/test/s2/librimix_as_array.npy')

# Löschen der zero-Einträge von s1
all_zero_rows_mask_s1 = np.all(s1 == 0, axis=1)
mix_clean = mix_clean[~all_zero_rows_mask_s1]
s1 = s1[~all_zero_rows_mask_s1]
s2 = s2[~all_zero_rows_mask_s1]

# Löschen der zero-Einträge von s2
all_zero_rows_mask_s2 = np.all(s2 == 0, axis=1)
mix_clean = mix_clean[~all_zero_rows_mask_s2]
s1 = s1[~all_zero_rows_mask_s2]
s2 = s2[~all_zero_rows_mask_s2]

# Abspeichern der ndarrays ohne Nullen
mix_clean_path = '/isip/Public/spruenken/LibriMix/0.25_without_zeros/min/test/mix_clean/'
s1_path = '/isip/Public/spruenken/LibriMix/0.25_without_zeros/min/test/s1/'
s2_path = '/isip/Public/spruenken/LibriMix/0.25_without_zeros/min/test/s2/'

# mix_clean
if not os.path.exists(mix_clean_path):
    os.makedirs(mix_clean_path)
np.save(os.path.join(mix_clean_path, 'librimix_as_array.npy'), mix_clean)
#s1
if not os.path.exists(s1_path):
    os.makedirs(s1_path)
np.save(os.path.join(s1_path, 'librimix_as_array.npy'), s1)
#s2
if not os.path.exists(s2_path):
    os.makedirs(s2_path)
np.save(os.path.join(s2_path, 'librimix_as_array.npy'), s2)