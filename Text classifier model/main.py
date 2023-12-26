from data.instructions_form import instructions_form
from data.instructions_browser import instructions_browser
from data.instructions_wait import instructions_wait
from data.instructions_not import instructions_not
from keras.preprocessing.text import Tokenizer
from text_classification_model_implements import TextClassificationModelImplements

def main():
    # Uso del modelo
    clas = [0, 1, 2, 3]
    model = TextClassificationModelImplements(clas, Tokenizer(),instructions_browser + instructions_form + instructions_wait + instructions_not)
    model.build_model()
    # Entrenamiento del modelo
    data_training = [instructions_browser,instructions_form,instructions_wait,instructions_not]
    label_training = [0, 1, 2, 3]  # Las etiquetas deben coincidir con las clases definidas en clases
    model.train_model(data_training, label_training, epochs=1000)  # Aumentar epochs

    # Ejemplo de predicción con el modelo
    new_instruction = "Espera 4 segundos antes de proceder con la siguiente acción"
    probabilidades = model.predict_probability(new_instruction)
    print("Probabilidades predichas para cada clase:")
    print(probabilidades)
    model.plot_training_history()

    #"Ingresa los siguientes datos en el formulario: Modelo 'Samsung Galaxy Watch 4', Color 'Dorado', Almacenamiento '32GB', Precio '249.99 USD', Tienda 'GizmoHub'"
    #"Navega hacia https://www.paginanueva.com usando Firefox"
    #"Espera 4 segundos antes de proceder con la siguiente acción"
    
if __name__ == "__main__":
    main()

'''
Redes neuronales recurrentes (RNN) con capas LSTM (Long Short-Term Memory)

Curva de Pérdida (Loss Curve):

Eje X: Épocas (número de veces que el modelo ha visto todo el conjunto de entrenamiento).
Eje Y: Pérdida del modelo en el conjunto de entrenamiento (loss).
Línea Azul: Representa la pérdida durante el entrenamiento.
Línea Naranja: Representa la pérdida durante la validación.

Curva de Precisión (Accuracy Curve):

Eje X: Épocas.
Eje Y: Precisión del modelo en el conjunto de entrenamiento y validación.
Línea Azul: Precisión durante el entrenamiento.
Línea Naranja: Precisión durante la validación.

loss - perdida
'''