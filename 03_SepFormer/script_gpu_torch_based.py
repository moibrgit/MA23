import subprocess as sp

import os
from multiprocessing import Process, Pipe

#os.environ["CUDA_VISIBLE_DEVICES"] ='0'
import numpy as np
import pynvml
import sys
def mask_unused_gpus(leave_unmasked=1, needed_memory= 5000, proc_id=0):
    def masking_process(child_conn, leave_unmasked,needed_memory, proc_id):
        try:
            # from tensorflow.python.client import device_lib
            # device_lib.list_local_devices()
            import torch
            num_of_gpus=torch.cuda.device_count()
            memory_free_values=[0] * num_of_gpus
            for i in range(num_of_gpus):
                try:
                    memory_free_values[i] = int(np.ceil(torch.cuda.mem_get_info(i)[0]/(10**6)))
                except:
                    memory_free_values[i] = 0
            print(memory_free_values)
            open_slots = (np.array(memory_free_values)/needed_memory).astype(int)
            available_gpus = [i for i, x in enumerate(np.cumsum(open_slots))if x > proc_id]
            #available_gpus = [i for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]

            if len(available_gpus) < leave_unmasked: raise ValueError(
                'Found only %d usable GPUs in the system' % len(available_gpus))


            child_conn.send(available_gpus)
            return available_gpus[:leave_unmasked]

        except Exception as e:

            print('"nvidia-smi" is probably not installed. GPUs are not masked', e)

    print('Masking GPUs with less than ' + str(needed_memory) + ' MB free VRAM')
    parent_conn, child_conn = Pipe()
    p = Process(target=masking_process, args=(child_conn,leave_unmasked, needed_memory, proc_id,))
    p.start()
    available_gpus = parent_conn.recv()
    p.join()
    print('Leaving GPU(s) ' + ','.join(map(str, available_gpus[:leave_unmasked])) + ' visible.')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, available_gpus[:leave_unmasked]))
    return None


def tf_set_memory_usage_dynamic():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def tf_set_memory_usage_limit(memory_limit=5000):
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

if __name__ == '__main__':
    import script_gpu_torch_based
    script_gpu_torch_based.mask_unused_gpus(needed_memory=5000)
    script_gpu_torch_based.tf_set_memory_usage_dynamic()
    #/isip/Public/pallenberg/venv/sepformerTaylorTFPytorch/bin/python3.10 
  