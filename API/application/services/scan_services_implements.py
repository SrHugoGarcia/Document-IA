from http import HTTPStatus
import re
from time import sleep
import logging
from datetime import datetime
import os

from constants.sentences import sentences_scan_public, elementos_dom, browsers, login, element_clicks
from exceptions.instruction_not_found_exception import InstructionNotFoundException
from exceptions.invalid_sentence_exception import InvalidSentenceException
from services.selenium_services_implements import SeleniumServicesImplements
from utils.selenium_config import SeleniumConfig
from utils.text_analyzer import TextAnalyzer
from utils.xpath_processor import XPathProcessor
from utils.file_manager import FileManager
from .scan_services import ScanServices

class ScanServicesImplements(ScanServices):
    def __init__(self):
        self._text_analyzer = TextAnalyzer()
        self._selenium_config = SeleniumConfig()
    
    def process_and_scan_elements(self, instruction: str):
        if not instruction:
            logging.error("La instrucción no puede estar vacía.")
            raise InstructionNotFoundException(message="La instrucción no puede estar vacía", error="Instrucción inválida", status_code=HTTPStatus.NOT_FOUND)
        
        if not self.__validate_sentence(instruction, sentences_scan_public):
            logging.error("La estructura de la oración es inválida.")
            raise  InvalidSentenceException(message="Oración inválida", error="La oración tiene un formato inválido", status_code=HTTPStatus.BAD_REQUEST)

        logging.info("Iniciando el análisis de texto para extraer información relevante.")
        result = self._text_analyzer.analyze_text(instruction=instruction, entity_config={
            'url': {'triggers': ['web'], 'allowed_pos': {'NOUN','PRON','SYM','PROPN'}},
            'browser_type': {'triggers': ['inicia en'], 'allowed_pos': {'NOUN'}},
            'element_dom_type': {'triggers': ['identificado como'], 'allowed_pos': {"NOUN", "PROPN", "ADJ", "CCONJ", "VERB", "ADP", "INTJ", "ADV", "PRON"}},
        })
  
        fields = ['url', 'browser_type', 'element_dom_type']
        invalid_fields = {field: result[field] for field in fields if not result[field]}
        # Filtra los campos que están ausentes o son nulos en el resultado
        if invalid_fields:
            self.__handle_invalid_sentence_fields(invalid_fields)
            
        self.__check_and_raise_invalid_element_types(element_types=[result['element_dom_type']],elements=elementos_dom)
 
        self.__check_and_raise_invalid_browser(browser_type=result['browser_type'])
        
        try:
            logging.info("Configurando y utilizando Selenium para obtener información de la página web.")
            driver = self._selenium_config.get_browser_instance(result['browser_type'].lower())
            self._selenium_services = SeleniumServicesImplements(driver)
            self._selenium_services.open_url(result['url'])
            self.xpath_elements = self._selenium_services.generate_xpaths_for_elements(tag_name_to_search=result['element_dom_type'])
            self._selenium_services.close_browser()
            
            # Construir la ruta de destino
            destination_folder = os.getcwd()+"/application/data"

            # Se le envia la información al file manager la ruta y el nombre del archivo
            FileManager(directory=destination_folder+f'/scan-public-{self.__get_current_date_in_desired_format()}.txt').write_content(self.xpath_elements)       
            
            
        except Exception as e:
            logging.error(f"Error durante la ejecución de Selenium: {e}")
            self._selenium_services.close_browser()
            raise e

        logging.info("Elementos escaneados devueltos en forma de XPath.")
        return self.xpath_elements
    

    
    def login_and_scan_elements(self, instruction: str):
        if not instruction:
            logging.error("La instrucción no puede estar vacía.")
            raise InstructionNotFoundException(message="La instrucción no puede estar vacía", error="Instrucción inválida", status_code=HTTPStatus.NOT_FOUND)
        
        if not self.__validate_sentence(instruction, login):
            logging.error("La estructura de la oración para iniciar sesión es inválida.")
            raise InvalidSentenceException(message="Oración inválida", error="La oración tiene un formato inválido", status_code=HTTPStatus.BAD_REQUEST)
        
        result = self._text_analyzer.analyze_text(instruction=instruction, entity_config={
            'url': {'triggers': ["web"], 'allowed_pos': {'NOUN','PRON','SYM','PROPN'}},
            'browser_type': {'triggers': ['inicia en'], 'allowed_pos': {'NOUN'}},
            'element_dom_type': {'triggers': ['designado como'], 'allowed_pos': {"NOUN", "PROPN", "ADJ", "CCONJ", "VERB", "ADP", "INTJ", "ADV", "PRON"}},
            'text_of_element_dom_type': {'triggers': ['la cadena'], 'allowed_pos': {"NOUN", "PROPN", "ADJ", "CCONJ", "VERB", "ADP", "INTJ", "ADV", "PRON"}},
            'click_element_dom_type': {'triggers': ['clic en el'], 'allowed_pos': {"NOUN", "PROPN", "ADJ", "CCONJ", "VERB", "ADP", "INTJ", "ADV", "PRON"}},
            'text_of_click': {'triggers': ['el texto'], 'allowed_pos': {"NOUN", "PROPN", "ADJ", "CCONJ", "VERB", "ADP", "INTJ", "ADV", "PRON"}},
            'element_dom_type_dash': {'triggers': ['identificado como'], 'allowed_pos': {"NOUN", "PROPN", "ADJ", "CCONJ", "VERB", "ADP", "INTJ", "ADV", "PRON"}},
            'text_of_element_dom_type_dash': {'triggers': ['la palabra'], 'allowed_pos': {"NOUN", "PROPN", "ADJ", "CCONJ", "VERB", "ADP", "INTJ", "ADV", "PRON"}},
            'element_scan': {'triggers': ['elementos de'], 'allowed_pos': {"NOUN", "PROPN", "ADJ", "CCONJ", "VERB", "ADP", "INTJ", "ADV", "PRON"}},
        })

        fields = ['url', 'browser_type', 'element_dom_type', 'text_of_element_dom_type', 'click_element_dom_type', 'text_of_click', 'element_dom_type_dash', 'text_of_element_dom_type_dash', 'element_scan']
        invalid_fields = {field: result[field] for field in fields if not result[field]}
        
        # Filtra los campos que están ausentes o son nulos en el resultado
        if invalid_fields:
            self.__handle_invalid_sentence_fields(invalid_fields)

        self.__check_and_raise_invalid_element_types(element_types=[result['element_dom_type'],result['element_dom_type_dash'],result['element_scan']],elements=elementos_dom)
        
        self.__check_and_raise_invalid_browser(browser_type=result['browser_type'])

        try:
            logging.info("Configurando y utilizando Selenium para obtener información de la página web.")
            driver = self._selenium_config.get_browser_instance(result['browser_type'].lower())
            self._selenium_services = SeleniumServicesImplements(driver)
            self._selenium_services.open_url(result['url'])

            xpath_elements = self._selenium_services.generate_xpaths_for_elements(tag_name_to_search=result['element_dom_type'])
            self.__check_and_raise_invalid_text(text=result['text_of_element_dom_type'],elements=xpath_elements)
                
            inputs = self._selenium_services.generate_xpaths_for_elements(tag_name_to_search='input')
            regex_pattern = r"'([^']+)' se completa con el valor '([^']+)'"
            if not XPathProcessor.validate_xpath_completion(instruction=instruction, elements=inputs, regex_pattern=regex_pattern):
                logging.error("Campo inválido, algún campo es inválido.")
                raise InvalidSentenceException(
                    message="Campo inválido, no existe el texto en la pagina",
                    error="Campo inválido",
                    status_code=HTTPStatus.BAD_REQUEST
                )

            self.__validate_generate_xpaths_for_element(click_element_dom_type=result['click_element_dom_type'])
            
            self.__check_and_raise_invalid_text(text=result['text_of_click'],elements=self.elements)
                
            send_keys = XPathProcessor.filter_elements_by_xpath_and_values(instruction=instruction, elements=inputs, regex_pattern=regex_pattern)
            sleep(3)
            self._selenium_services.send_keys_dynamic(send_keys)
            sleep(3)
            self._selenium_services.click_element(element=self.elements[0])
            sleep(6)
            logging.info("0la")
            logging.info(result['element_dom_type_dash'])
            xpath_elements_dash = self._selenium_services.generate_xpaths_for_elements(tag_name_to_search=result['element_dom_type_dash'])
            logging.info("0")

            self.__check_and_raise_invalid_text(text=result['text_of_element_dom_type_dash'],elements=xpath_elements_dash)
            logging.info("1")
            self.scan = self._selenium_services.generate_xpaths_for_elements(tag_name_to_search=result['element_scan'])
            logging.info("2")
            #cookies = driver.get_cookies()
            #local_storage = driver.execute_script("return window.localStorage;")
            #session_storage = driver.execute_script("return window.sessionStorage;")
            #print(cookies)
            #print(local_storage)
            #print(session_storage)
            self._selenium_services.close_browser()
            #FileManager(directory=f"{os.getcwd()}\\application\\data\\scan-private-{self.__get_current_date_in_desired_format()}.txt").write_content(self.scan)
            # Construir la ruta de destino
            destination_folder = os.getcwd()+"/application/data"

            # Se le envia la información al file manager la ruta y el nombre del archivo
            FileManager(directory=destination_folder+f'/scan-private-{self.__get_current_date_in_desired_format()}.txt').write_content(xpath_elements)   

        except Exception as e:
            logging.error(f"Error durante la ejecución de Selenium: {e}")
            self._selenium_services.close_browser()
            raise e

        logging.info("Elementos escaneados devueltos en forma de XPath.")
        return self.scan
    
    
    def process_and_scan_elements_v2(self, url: str,tag: str,browser: str):
        if not url:
            raise InstructionNotFoundException(message="La url no puede estar vacio", error="Url invalida", status_code=HTTPStatus.NOT_FOUND)
        
        if not tag:
            raise InstructionNotFoundException(message="El tag no puede estar vacio", error="Tag invalido", status_code=HTTPStatus.NOT_FOUND)
        
        if not browser:
            raise InstructionNotFoundException(message="El navegador no puede estar vacio", error="Navegador invalido", status_code=HTTPStatus.NOT_FOUND)
        
        #self.__check_and_raise_invalid_element_types(element_types=[tag],elements=elementos_dom)
        self.__check_and_raise_invalid_browser(browser_type=browser)
        xpath_elements = []
        try:
            logging.info("Configurando y utilizando Selenium")
            driver = self._selenium_config.get_browser_instance(browser)
            self._selenium_services = SeleniumServicesImplements(driver)
            self._selenium_services.open_url(url)
            logging.info(self._selenium_services.get_all_tags())
            if tag == "all_tags":
                for element in self._selenium_services.get_all_tags():
                    logging.info(element)
                    for e in self._selenium_services.generate_xpaths_for_elements(tag_name_to_search=element):
                        xpath_elements.append(e)
            else:
                xpath_elements =self._selenium_services.generate_xpaths_for_elements(tag_name_to_search=tag)
            self._selenium_services.close_browser()
            # Construir la ruta de destino
            destination_folder = os.getcwd()+"/application/data"

            # Se le envia la información al file manager la ruta y el nombre del archivo
            FileManager(directory=destination_folder+f'/scan-public-{self.__get_current_date_in_desired_format()}.txt').write_content(xpath_elements)   
        except Exception as e:
            logging.error(f"Error durante la ejecución de Selenium: {e}")
            self._selenium_services.close_browser()
            raise e
        logging.info("Elementos escaneados devueltos en forma de XPath.")
        return xpath_elements


    # Método para validar la estructura de una oración
    def __validate_sentence(self, instruction, sentences):
        for pattern in sentences:
            match = re.search(pattern, instruction)
            if not match:
                return False
        return True


    ## Método para buscar una palabra en una lista de diccionarios
    def __search_word(self, word, dictionary_list):
        for dictionary in dictionary_list:
            if 'text' in dictionary and dictionary['text'] is not None and word in dictionary['text']:
                return True
        return False


    # Método para validar y generar XPath para el elemento de clic
    def __validate_generate_xpaths_for_element(self, click_element_dom_type):
        if click_element_dom_type in element_clicks:
            self.elements = self._selenium_services.generate_xpaths_for_elements(tag_name_to_search=click_element_dom_type)
        else:
            raise InvalidSentenceException(
                message=f"Invalid element: {click_element_dom_type} not allowed.",
                error="Invalid element",
                status_code=HTTPStatus.BAD_REQUEST
            )

    # Método para manejar campos inválidos en la estructura de la oración
    def __handle_invalid_sentence_fields(self, invalid_fields):
        # Registra un mensaje de error indicando los campos inválidos
        logging.error(f"Campos inválidos: {', '.join(invalid_fields.keys())}")
        
        # Registra el contenido específico de los campos inválidos
        logging.error(f"Contenido de campos inválidos: {invalid_fields}")
        
        # Lanza una excepción específica indicando que la estructura de la oración es inválida
        raise InvalidSentenceException(
            message="Oración inválida",
            error=f"Los siguientes campos son inválidos: {', '.join(invalid_fields.keys())}",
            status_code=HTTPStatus.BAD_REQUEST
        )
    
    # Verifica si el texto no se encuentra en la lista de elementos y genera una excepción si es así.
    def __check_and_raise_invalid_text(self, text, elements):
        if not self.__search_word(text, elements):
            logging.error(f"No se encontró el texto: {text}")
            raise InvalidSentenceException(
                message=f"No se encontró el texto: {text}",
                error="Texto inválido",
                status_code=HTTPStatus.BAD_REQUEST
            )
            
    # Verifica si el tipo de navegador no es válido y genera una excepción si es así.
    def __check_and_raise_invalid_browser(self, browser_type):
        if browser_type not in browsers:
            logging.error("Tipo de navegador inválido.")
            raise InvalidSentenceException(
                message="Tipo de navegador inválido",
                error="El navegador tiene un formato inválido",
                status_code=HTTPStatus.BAD_REQUEST
            )
    
    # Verifica si alguno de los tipos de elementos no es válido y genera una excepción si es así.
    def __check_and_raise_invalid_element_types(self, element_types, elements):
        invalid_elements = [element for element in element_types if element not in elements]
        if invalid_elements:
            invalid_elements_str = ', '.join(invalid_elements)
            logging.error(f"Tipo de elemento inválido: {invalid_elements_str}")
            raise InvalidSentenceException(
                message=f"Tipo de elemento inválido: {invalid_elements_str}",
                error="El tipo de elemento tiene un formato inválido",
                status_code=HTTPStatus.BAD_REQUEST
            )

    def __get_current_date_in_desired_format(self):
            # Obtener la fecha y hora actual
            current_datetime = datetime.now()

            # Formatear la fecha y hora en un formato deseado, por ejemplo, 'YYYY-MM-DD_HH-MM-SS'
            formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
            # Construir el nombre del archivo con la fecha y hora formateadas
            return f"{formatted_datetime}"