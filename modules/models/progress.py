import tensorflow as tf

from modules.utilities import logging


class ProgressMonitor(tf.keras.callbacks.Callback):
    """
    Custom Keras callback to monitor training progress and implement early stopping.

    Args:
        patience (int): Number of epochs to wait for improvement before stopping training.
        min_delta (float): Minimum change to qualify as an improvement.
    """

    def __init__(self, patience=5, min_delta=1e-5):
        super(ProgressMonitor, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.no_improvement_count = 0

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to check for improvement.

        Args:
            epoch (int): The current epoch number.
            logs (dict): The logs from the current epoch.
        """
        current_loss = logs.get('val_loss')
        if current_loss is None:
            return

        if (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        if self.no_improvement_count >= self.patience:
            logging.info(f"Stopping training due to lack of improvement after {self.patience} epochs")
            self.model.stop_training = True

    def on_train_batch_end(self, batch, logs=None):
        """
        Called at the end of each training batch to log progress.

        Args:
            batch (int): The current batch number.
            logs (dict): The logs from the current batch.
        """
        if batch % 10 == 0:  # Log every 10 batches
            logging.info(f"Batch {batch}: loss = {logs['loss']:.6f}")
