import subprocess as sp

import os


def mask_unused_gpus(leave_unmasked=1):
    ACCEPTABLE_AVAILABLE_MEMORY = 5000

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"

    print('Masking GPUs with less than ' + str(ACCEPTABLE_AVAILABLE_MEMORY) + ' MB free VRAM')

    try:

        _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]

        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]

        print(memory_free_values)

        available_gpus = [i for i, x in enumerate(memory_free_values) if x > ACCEPTABLE_AVAILABLE_MEMORY]

        if len(available_gpus) < leave_unmasked: raise ValueError(
            'Found only %d usable GPUs in the system' % len(available_gpus))

        print('Leaving GPU(s) ' + ','.join(map(str, available_gpus[:leave_unmasked])) + ' visible.')

        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, available_gpus[:leave_unmasked]))

        return available_gpus

    except Exception as e:

        print('"nvidia-smi" is probably not installed. GPUs are not masked', e)

    return None
