from data.instructions_form import instructions_form
from data.instructions_browser import instructions_browser
from data.instructions_wait import instructions_wait
from data.instructions_not import instructions_not
from keras.preprocessing.text import Tokenizer
from text_classification_rnn_model import TextClassificationRNNModel
from text_classification_feedforward_model import TextClassificationFeedforwardModel
from text_classification_modular_model import TextClassificationModularModel
def main():
    # Uso del modelo
    clas = [0, 1, 2, 3]
    model = TextClassificationModularModel(clases=clas, tokenizer=Tokenizer(),instruction=instructions_browser + instructions_form + instructions_wait + instructions_not)
    model.build_model()
    # Entrenamiento del modelo
    data_training = [instructions_browser,instructions_form,instructions_wait,instructions_not]
    
    x_train, x_val, y_train, y_val = model.prepare_data(data_training_class=data_training)
    model.train_model(x_train, y_train, x_val, y_val)

    # Ejemplo de predicción con el modelo
    new_instruction =  "Abre el navegador Edge y ve a https://www.paginanueva.com"
    model.predict_probability(new_instruction)

    new_instruction_II =  "Ingresa los siguientes datos en el formulario: Modelo 'Samsung Galaxy Watch 4', Color 'Dorado', Almacenamiento '32GB', Precio '249.99 USD', Tienda 'GizmoHub'"
    model.predict_probability(new_instruction_II)
    
    new_instruction_III =  "Espera 4 segundos antes de proceder con la siguiente acción"
    model.predict_probability(new_instruction_III)
    
    new_instruction_IV =  "Esto es una prueba"
    model.predict_probability(new_instruction_IV)
    
    new_instruction_IV =  "Utiliza Firefox para acceder a https://www.paginanueva.com"
    model.predict_probability(new_instruction_IV)
    
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