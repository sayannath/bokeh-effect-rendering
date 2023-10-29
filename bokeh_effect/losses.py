import tensorflow as tf

def peak_signal_noise_ratio(y_true, y_pred) -> tf.float32:
    return tf.image.psnr(y_pred, y_true, max_val=255.0)