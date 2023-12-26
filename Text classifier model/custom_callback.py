from keras.callbacks import EarlyStopping
import tensorflow as tf

class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None,min_accuracy=0.95):
        # Verificar si se cumple la condición de precisión
        if logs['accuracy'] >= min_accuracy and logs['val_accuracy'] >= min_accuracy:
            print(f"Alcanzado el accuracy mínimo deseado ({min_accuracy}). Deteniendo el entrenamiento.")
            self.model.stop_training = True