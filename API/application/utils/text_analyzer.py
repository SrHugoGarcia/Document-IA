import spacy
import logging

class TextAnalyzer:
    def __init__(self):
        # Cargar el modelo de procesamiento de lenguaje natural de Spacy para español
        self.nlp = spacy.load("es_core_news_sm")
        logging.info("Modelo de procesamiento de lenguaje natural cargado.")

    def analyze_text(self, instruction, entity_config):
        logging.info("Iniciando el analisis del texto")
        # Procesar la instrucción utilizando Spacy
        doc = self.nlp(instruction)
        
        # Inicializar el diccionario de resultados con cadenas vacías para cada entidad
        result = {entity: '' for entity in entity_config.keys()}

        # Iterar sobre cada entidad y su configuración
        for entity, config in entity_config.items():
            logging.info(f"Analizando entidad: {entity}")
            # Obtener desencadenadores y partes del discurso permitidas para la entidad
            triggers = config['triggers']
            allowed_pos = config['allowed_pos']

            # Iterar sobre cada token en el documento
            for i, token in enumerate(doc):
                # Iterar sobre cada desencadenador para la entidad actual
                for trigger in triggers:
                    # Dividir el desencadenador en tokens individuales
                    trigger_tokens = trigger.split()
                    # Verificar si la secuencia de tokens coincide con el desencadenador
                    if i + len(trigger_tokens) <= len(doc):
                        if all(doc[i + j].text.lower() == trigger_token.lower() for j, trigger_token in enumerate(trigger_tokens)):
                            #logging.info(f'Se encontró el desencadenador "{trigger}" para la entidad "{entity}" en el índice {i}')
                            start_idx = i + len(trigger_tokens)
                            inside_quotes = False  # Bandera para indicar si estamos dentro de comillas simples
                            
                            # Iterar sobre los tokens después del desencadenador
                            while start_idx < len(doc) and (
                                    doc[start_idx].pos_ in allowed_pos or doc[start_idx].text == "'" or doc[start_idx].text == '"'):
                                # Verificar si el token actual cumple con los criterios permitidos
                                if doc[start_idx].text == "'":
                                    inside_quotes = not inside_quotes  # Cambiar la bandera al encontrar comillas simples
                                elif inside_quotes and doc[start_idx].text not in ["'", '"'] and doc[start_idx].pos_ in allowed_pos:
                                    
                                    #logging.info(f'Desencadenador: "{trigger}" | Palabra Actual: "{doc[start_idx].text}"')
                                    # Agregar la palabra al resultado de la entidad actual
                                    if doc[start_idx].text.lower() != trigger.lower():

                                        #logging.info(f'Añadiendo la palabra "{doc[start_idx].text}" a la entidad "{entity}"')
                                        result[entity] += doc[start_idx].text.replace("'", '') + ' '
                                start_idx += 1

                            # Limpiar y actualizar el resultado para la entidad actual
                            result[entity] = result[entity].strip()
                            break

        # Devolver el diccionario final de resultados
        logging.info("Análisis del texto completado.")
        return result

