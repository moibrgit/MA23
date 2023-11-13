import tensorflow as tf

class SiSNRLoss(tf.keras.losses.Loss):
    # def __init__(self, name="sisnr_loss", **kwargs):

    def __init__(self, ):
        super().__init__()

    def call(self, y_true, y_pred):
        """
        Compute the Scale-Invariant Signal-to-Noise Ratio (SiSNR) loss.

        Parameters:
        y_true (tf.Tensor): The ground truth tensor, shape expected to be (batch_size, num_speakers, sequence_length, 1).
        y_pred (tf.Tensor): The predicted tensor, shape expected to be (batch_size, num_speakers, sequence_length, 1).
        
        Returns:
        tf.Tensor: The SiSNR loss.
        """
        

        print("y_true data type:", y_true.dtype)
        print("y_pred data type:", y_pred.dtype)
        
        print("y_true:", y_true)
        print("y_pred:", y_pred)


        # Remove the channel dimension (assumed to be 1)
        y_true = tf.squeeze(y_true, -1)
        y_pred = tf.squeeze(y_pred, -1)

       

        # Mean subtraction
        mean_true = tf.reduce_mean(y_true, axis=2, keepdims=True)
        mean_pred = tf.reduce_mean(y_pred, axis=2, keepdims=True)
        y_true_zm = y_true - mean_true
        y_pred_zm = y_pred - mean_pred

        # SiSNR calculation
        dot_product = tf.reduce_sum(y_true_zm * y_pred_zm, axis=2, keepdims=True)
        norm_true = tf.reduce_sum(y_true_zm ** 2, axis=2, keepdims=True) + 1e-8
        scale = dot_product / norm_true
        projected = scale * y_true_zm
        noise = y_pred_zm - projected
        ratio = tf.reduce_sum(projected ** 2, axis=2) / (tf.reduce_sum(noise ** 2, axis=2) + 1e-8)
        sisnr = 10 * tf.math.log(ratio + 1e-8) / tf.math.log(10.0)

        # Since SiSNR is typically maximized, return the negative mean SiSNR for minimization
        return -tf.reduce_mean(sisnr)

# Example usage with TensorFlow model:
# model.compile(optimizer='adam', loss=SiSNRLoss())
