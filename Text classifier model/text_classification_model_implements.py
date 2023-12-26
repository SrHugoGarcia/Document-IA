import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from text_classification_model import TextClassificationModel
from custom_callback import CustomCallback

class TextClassificationModelImplements(TextClassificationModel):
    def __init__(self, clases=[], tokenizer=None,instruction = []):
        # Inicialización del modelo y configuración de capas
        self.clases = clases
        self.tokenizer = tokenizer
        self.tokenizer.fit_on_texts(instruction)
        

    def build_model(self):
        self.model = Sequential([
            Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=64, input_length=50),  #Tamaño del vocabulario(palabras unicas),salida, longitud máxima de las oraciones que se alimentarán a la red. 
            LSTM(units=150, return_sequences=True, kernel_regularizer=l2(0.01)),  #capturar patrones a largo plazo en los datos, Aumentar unidades(patrones) y agregar regularización
            Dropout(0.5),  # Aumentar Dropout
            LSTM(units=150, kernel_regularizer=l2(0.01)),  # Aumentar unidades y agregar regularización
            Dense(units=256, activation='relu', kernel_regularizer=l2(0.01)),  # cada neurona o unidad está conectada a cada neurona de la capa anterior.
            BatchNormalization(), #Tecnica estabilizar y acelerar el aprendizaje.
            Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)),  # Cambiar activación y agregar regularización
            Dropout(0.5),  # Ajustar Dropout, técnica de regularización que se utiliza para prevenir el sobreajuste en redes neuronales.
            Dense(units=len(self.clases), activation='softmax')
        ])
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, data_training_class, label_training_class, epochs=20):  # Aumentar epochs
        # Preparación de datos para el entrenamiento
        all_texts = []
        all_labels = []

        for i, clase in enumerate(self.clases):

            texts = data_training_class[i]
            labels = [i] * len(texts)
            all_texts.extend(texts)
            all_labels.extend(labels)

        # Tokenización y secuenciación de textos
        if self.tokenizer is None:
            self.tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
            self.tokenizer.fit_on_texts(all_texts)
        sequences = self.tokenizer.texts_to_sequences(all_texts)
        padded_sequences = pad_sequences(sequences, maxlen=50, truncating='post')

        # División de datos en conjuntos de entrenamiento y validación
        x_train, x_val, y_train, y_val = train_test_split(padded_sequences, np.array(all_labels), test_size=0.3, random_state=54)

        # Convertir etiquetas a one-hot encoding
        y_train_numeric = np.array(y_train, dtype=int)
        y_train_categorical = tf.keras.utils.to_categorical(y_train_numeric - np.min(y_train_numeric), num_classes=len(self.clases))
        y_val_categorical = tf.keras.utils.to_categorical(y_val - min(y_val), num_classes=len(self.clases))

        # Entrenamiento del modelo
        print("Entrenamiento del Modelo:")
        #custom_callback = CustomCallback()
        #EarlyStopping 
        #se detendrá el entrenamiento si la pérdida en el conjunto de validación no mejora después de n épocas (patience=3)
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Guarda el mejor modelo durante el entrenamiento
        model_checkpoint_callback = ModelCheckpoint('best_model.h5', save_best_only=True)
        
        self.history = self.model.fit(
            x_train, y_train_categorical,
            epochs=epochs,
            validation_data=(x_val, y_val_categorical),
            verbose=2,
            callbacks=[model_checkpoint_callback,early_stopping_callback]
        )
        # Imprimir métricas detalladas durante el entrenamiento
        print("Métricas durante el entrenamiento:")
        print(self.history.history)

    def predict_probability(self, texto):
        # Preprocesamiento de texto para predicción
        sequence = self.tokenizer.texts_to_sequences([texto])
        padded_sequence = pad_sequences(sequence, maxlen=50, truncating='post')

        # Predicción de probabilidades
        probabilities = self.model.predict(np.array(padded_sequence))
        probability_for_class = probabilities[0]

        return probability_for_class

    def plot_training_history(self):
        # Gráficos de pérdida
        plt.plot(self.history.history['loss'], label='Entrenamiento')
        plt.plot(self.history.history['val_loss'], label='Validación')
        plt.title('Curva de Pérdida')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.legend()
        plt.show()

        # Gráficos de precisión
        plt.plot(self.history.history['accuracy'], label='Entrenamiento')
        plt.plot(self.history.history['val_accuracy'], label='Validación')
        plt.title('Curva de Precisión')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.legend()
        plt.show()
