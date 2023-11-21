import sys
sys.path.extend(['/home/spruenken/Documents/Thesis'])
import numpy as np
import yaml
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import src.tinySepformer as ts
from src.Sepformer import Sepformer


if __name__ == '__main__':
    # Parameters
    batch_size = 1

    for r in [444, 288]:
        if r == 288:
            #ts.intraCA_generel_nTimes = 8
            #ts.interCA_generel_nTimes = 8
            #ts.nmask_ntimes = 2
            #ts.caBlock_parameterSharing = False
            anzahl_k = 8
            anzahl_layer = 2
        else:
            #ts.intraCA_generel_nTimes = 4
            #ts.interCA_generel_nTimes = 4
            #ts.nmask_ntimes = 4
            #ts.caBlock_parameterSharing = True
            anzahl_k = 4
            anzahl_layer = 4

        for s in [8000, 16000, 24000, 32000, 40000, 48000, 56000, 64000]:  # 8000, 16000, 24000, 32000, 40000, 48000, 56000, 64000
            #ts.wav_vector_temp = s
            model = Sepformer(k=anzahl_k, num_layer=anzahl_layer)
            test_dataset = tf.random.uniform((100, s, 1), dtype=tf.float64)
            test_dataset = tf.data.Dataset.from_tensor_slices((test_dataset, test_dataset))
            test_dataset = test_dataset.batch(batch_size)

            model.predict(tf.random.uniform((1, s, 1), dtype=tf.float64), verbose=0)

            print('Für Modell', r, 'und Eingabelänge', s)
            tf.config.experimental.reset_memory_stats('GPU:0')
            print('Grundwert', tf.config.experimental.get_memory_info('GPU:0'))
            # Calculate the model
            model.predict(test_dataset)
            print('Nach Berechnung', tf.config.experimental.get_memory_info('GPU:0'))
            print('')
