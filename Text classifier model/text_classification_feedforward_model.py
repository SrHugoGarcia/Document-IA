from keras.models import Sequential
from keras.layers import Input, Embedding, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # Agregamos la importación de TensorFlow
from text_classification_model import TextClassificationModel
from keras.models import Model

class TextClassificationFeedforwardModel(TextClassificationModel):  # Asumiendo que TextClassificationModel tiene las funciones necesarias
    def __init__(self, clases=[], tokenizer=None, instruction=[]):
        self.clases = clases
        self.tokenizer = tokenizer
        self.tokenizer.fit_on_texts(instruction)

    def build_model(self):
        input_sequence = Input(shape=(50,))
        embedding_layer = Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=64, input_length=50)(input_sequence)
        flattened_layer = Flatten()(embedding_layer)
        dense_layer1 = Dense(units=150, activation='relu', kernel_regularizer=l2(0.01))(flattened_layer)
        dropout1 = Dropout(0.5)(dense_layer1)
        dense_layer2 = Dense(units=150, activation='relu', kernel_regularizer=l2(0.01))(dropout1)
        batch_norm1 = BatchNormalization()(dense_layer2)
        dense_layer3 = Dense(units=256, activation='relu', kernel_regularizer=l2(0.01))(batch_norm1)
        batch_norm2 = BatchNormalization()(dense_layer3)
        dense_layer4 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(batch_norm2)
        dropout2 = Dropout(0.5)(dense_layer4)
        output_layer = Dense(units=len(self.clases), activation='softmax')(dropout2)

        self.model = Model(inputs=input_sequence, outputs=output_layer)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


    def prepare_data(self, data_training_class):
        all_texts = []
        all_labels = []

        for i, clase in enumerate(self.clases):
            texts = data_training_class[i]
            labels = [i] * len(texts)
            all_texts.extend(texts)
            all_labels.extend(labels)

        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(all_texts)

        sequences = self.tokenizer.texts_to_sequences(all_texts)
        padded_sequences = pad_sequences(sequences, maxlen=50, truncating='post')

        x_train, x_val, y_train, y_val = train_test_split(padded_sequences, np.array(all_labels), test_size=0.3, random_state=54)
        y_train_numeric = np.array(y_train, dtype=int)
        y_train_categorical = tf.keras.utils.to_categorical(y_train_numeric - np.min(y_train_numeric), num_classes=len(self.clases))
        y_val_categorical = tf.keras.utils.to_categorical(y_val - np.min(y_val), num_classes=len(self.clases))

        return x_train, x_val, y_train_categorical, y_val_categorical

    def train_model(self, x_train, y_train, x_val, y_val, epochs=400):
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint_callback = ModelCheckpoint('best_model.h5', save_best_only=True)

        self.history = self.model.fit(
            x_train, y_train,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=2,
            callbacks=[model_checkpoint_callback, early_stopping_callback]
        )

        print("Métricas durante el entrenamiento:")
        print(self.history.history)

    def predict_probability(self, text):
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=50, truncating='post')
        probabilities = self.model.predict(np.array(padded_sequence))
        probability_for_class = probabilities[0]

        # Convertimos las probabilidades a porcentajes
        probabilities_percentage = probability_for_class * 100

        # Configuramos las opciones de impresión para evitar la notación científica
        np.set_printoptions(suppress=True, precision=4)

        print(f"Enunciado: {text}")
        print("Probabilidades en porcentaje para cada clase:")
        for i, percentage in enumerate(probabilities_percentage):
            print(f"Clase {i + 1}: {percentage:.4f}%")

        # Restauramos las opciones de impresión a su configuración original
        np.set_printoptions(suppress=False, precision=8)

        return probability_for_class


    def plot_training_history(self):
        plt.plot(self.history.history['loss'], label='Entrenamiento')
        plt.plot(self.history.history['val_loss'], label='Validación')
        plt.title('Curva de Pérdida')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.show()

        plt.plot(self.history.history['accuracy'], label='Entrenamiento')
        plt.plot(self.history.history['val_accuracy'], label='Validación')
        plt.title('Curva de Precisión')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()
        plt.show()
