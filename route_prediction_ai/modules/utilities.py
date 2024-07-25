import logging
import os
import platform
import signal
import threading
from pathlib import Path

import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_directory(directory):
    """
    Create a directory if it does not already exist.

    Args:
        directory (str): The path of the directory to create.
    """
    if not os.path.exists(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)


def timeout_handler(signum, frame):
    """
    Signal handler for function call timeouts.

    Args:
        signum: The signal number.
        frame: The current stack frame.
    """
    raise TimeoutError("Function call timed out")


def run_with_timeout(func, args=(), kwargs={}, timeout_duration=300):
    """
    Run a function with a timeout. If the function does not complete within the specified
    duration, a TimeoutError is raised.

    Args:
        func (callable): The function to run.
        args (tuple): Positional arguments to pass to the function.
        kwargs (dict): Keyword arguments to pass to the function.
        timeout_duration (int): The maximum time (in seconds) to allow the function to run.

    Returns:
        The result of the function call.
    """
    if platform.system() != 'Windows':
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_duration)
        try:
            result = func(*args, **kwargs)
        finally:
            signal.alarm(0)
    else:
        timer = threading.Timer(timeout_duration, timeout_handler, args=(None, None))
        timer.start()
        try:
            result = func(*args, **kwargs)
        finally:
            timer.cancel()
    return result


@tf.keras.utils.register_keras_serializable()
def scaled_mse(y_true, y_pred):
    """
    Custom scaled mean squared error loss function.

    Args:
        y_true (tf.Tensor): True target values.
        y_pred (tf.Tensor): Predicted values.

    Returns:
        tf.Tensor: Scaled MSE loss.
    """
    return tf.reduce_mean(tf.square((y_true - y_pred) * tf.constant([0.0001, 0.0001])))
