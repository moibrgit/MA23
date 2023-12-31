# Copyright (c) 2021 Chanjung Kim. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import math


class SiSNR(tf.keras.losses.Loss):
    """Implements the SiSNR function.

    Attributes:
        epsilon: A small constant for numerical stability
    """

    def __init__(self, epsilon: float = 1e-10):
        super(SiSNR, self).__init__()
        self.epsilon = epsilon

    def call(self, s, s_hat):
        s_target = s * (tf.reduce_sum(tf.multiply(s, s_hat)) /
                        tf.reduce_sum(tf.multiply(s, s)))
        e_noise = s_hat - s_target
        result = 20 * tf.math.log(tf.norm(e_noise) /
                                  (tf.norm(s_target + self.epsilon) + self.epsilon)) / math.log(10)
        return result


class SDR(tf.keras.losses.Loss):
    """Implements the SDR function.

    Attributes:
        epsilon: A small constant for enumerical stability
    """

    def __init__(self, epsilon: float = 1e-10, **kwargs):
        super(SDR, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, s, s_hat):
        # Ensure inputs are of type float64
        # FIXME: Error because s (original singal) is string and s_hat(estimated signal) ist float 64. I tried to cast, but didnt word
        # 
        print(f"S: {s}")
        print(f"S_hat: {s_hat}")

        s = tf.cast(s, tf.float64)
        s_hat = tf.cast(s_hat, tf.float64)

        # Check that inputs are not strings
        if not tf.is_tensor(s) or s.dtype.is_floating != True:
            raise TypeError("Input 's' must be a floating point tensor")
        if not tf.is_tensor(s_hat) or s_hat.dtype.is_floating != True:
            raise TypeError("Input 's_hat' must be a floating point tensor")

        return 20 * tf.math.log(tf.norm(s_hat - s) / (tf.norm(s) + self.epsilon) + self.epsilon) / math.log(10)
