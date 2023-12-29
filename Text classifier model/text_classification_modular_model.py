import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Embedding, LSTM, Dropout, Dense, BatchNormalization, Conv1D, GlobalMaxPooling1D, concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from text_classification_model import TextClassificationModel
from custom_callback import CustomCallback
from keras.models import Model
from keras.callbacks import Callback

class TextClassificationModularModel(TextClassificationModel):
    
    def __init__(self, clases=[], tokenizer=None, instruction=[]):
        self.clases = clases
        self.tokenizer = tokenizer
        self.tokenizer.fit_on_texts(instruction)

    '''
        Función: Esta función construye la capa de Embedding, que se utiliza para convertir secuencias de números enteros (índices de palabras) en vectores de longitud fija.
        Parámetros:
        input_sequence: La secuencia de entrada que representa las palabras.
        Capa Embedding:
        input_dim: Número de palabras únicas en el vocabulario más 1.
        output_dim: Dimensión de los vectores de embedding.
        input_length: Longitud de las secuencias de entrada.
    '''
    def _build_embedding_module(self, input_sequence):
        embedding_layer = Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=64, input_length=50)(input_sequence)
        return embedding_layer
    
    '''
        Función: Esta función construye un módulo LSTM (Long Short-Term Memory) para capturar patrones a largo plazo en los datos de secuencia.
        Parámetros:
        input_layer: La capa de entrada a la red.
        Capas LSTM:
        units: Número de unidades (neuronas) en la capa LSTM.
        return_sequences: Si es True, devuelve la secuencia completa.
        kernel_regularizer: Aplicación de regularización en los pesos.
        Dropout: Técnica de regularización para prevenir el sobreajuste.
    '''

    def _build_lstm_module(self, input_layer):
        lstm_layer1 = LSTM(units=150, return_sequences=True, kernel_regularizer=l2(0.01))(input_layer)
        lstm_dropout1 = Dropout(0.5)(lstm_layer1)
        lstm_layer2 = LSTM(units=150, kernel_regularizer=l2(0.01))(lstm_dropout1)
        return lstm_layer2
    
    '''
        Función: Construye un módulo convolucional para identificar patrones locales en secuencias.
        Parámetros:
        input_layer: Capa de entrada a la red.
        Capas Conv1D:
        filters: Número de filtros convolucionales.
        kernel_size: Tamaño de la ventana del kernel.
        activation: Función de activación (ReLU en este caso).
        GlobalMaxPooling1D: Reduce la dimensionalidad de la salida.
    '''
    def _build_convolutional_module(self, input_layer):
        conv_layer = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
        conv_pooling = GlobalMaxPooling1D()(conv_layer)
        return conv_pooling
    
    '''
        Función: Construye un módulo densamente conectado para aprendizaje no secuencial.
        Parámetros:
        input_layer: Capa de entrada a la red.
        Capas Dense:
        units: Número de unidades en la capa densa.
        activation: Función de activación (ReLU en este caso).
        kernel_regularizer: Aplicación de regularización en los pesos.
        BatchNormalization: Normalización de lotes para estabilizar el entrenamiento.
        Dropout: Técnica de regularización para prevenir el sobreajuste.
    '''
    def _build_dense_module(self, input_layer):
        dense_layer1 = Dense(units=256, activation='relu', kernel_regularizer=l2(0.01))(input_layer)
        dense_batch_norm = BatchNormalization()(dense_layer1)
        dense_layer2 = Dense(units=128, activation='relu', kernel_regularizer=l2(0.01))(dense_batch_norm)
        dense_dropout = Dropout(0.5)(dense_layer2)
        return dense_dropout

    def build_model(self):
        #Se define una capa de entrada que espera secuencias de longitud 50.
        input_sequence = Input(shape=(50,))

        # Módulo de Embedding
        #La secuencia de entrada se pasa a través del módulo de Embedding para convertir las palabras en vectores densos.
        embedding_output = self._build_embedding_module(input_sequence)

        # Módulo Recurrente (LSTM)
        #La salida del módulo de Embedding se pasa a través de un módulo LSTM para capturar patrones a largo plazo en la secuencia.
        lstm_output = self._build_lstm_module(embedding_output)

        # Módulo Convolucional
        #La salida del módulo de Embedding también se pasa a través de un módulo convolucional para capturar patrones locales en la secuencia.
        convolutional_output = self._build_convolutional_module(embedding_output)

        # Concatenar las salidas de los módulos
        #Las salidas del módulo LSTM y el módulo convolucional se concatenan para combinar información aprendida tanto a nivel local como a largo plazo.
        concatenated_output = concatenate([lstm_output, convolutional_output])

        # Módulo Densa
        #La salida concatenada se pasa a través de un módulo densamente conectado para realizar operaciones no secuenciales y aprender relaciones más complejas.
        dense_output = self._build_dense_module(concatenated_output)

        # Capa de Salida
        #la salida del módulo denso se conecta a una capa de salida que tiene tantas unidades como clases y utiliza la función de activación softmax para obtener probabilidades de clasificación.
        output_layer = Dense(units=len(self.clases), activation='softmax')(dense_output)

        # Crear el modelo completo
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
            callbacks=[model_checkpoint_callback,early_stopping_callback]
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
