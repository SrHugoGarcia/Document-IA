from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

from exceptions.element_not_found_exception import ElementNotFoundException
from http import HTTPStatus
from bs4 import BeautifulSoup

import logging
from time import sleep
from utils.scripts import script_path_absolute
from .selenium_services import SeleniumServices
from time import sleep
class SeleniumServicesImplements(SeleniumServices):
    def __init__(self, driver):
        self.driver = driver
        self.wait = WebDriverWait(self.driver, 10)

    ## Método para obtener un elemento mediante su estrategia de búsqueda y valor
    def get_element(self, by, value):
        try:
            return self.driver.find_element(by,value)
        except Exception as e:
           raise  e
       
    # Método para esperar la visibilidad de un elemento
    def wait_for_element(self, by, value):
        try:
            return self.wait.until(
                EC.visibility_of_element_located((by, value))
            )
        except Exception as e:
            raise e
    
    # Método para obtener la URL actual
    def get_current_url(self):
        try:
            return self.driver.current_url
        except Exception as e:
            raise e

    # Método para abrir una URL en el navegador
    def open_url(self,url):
        try:
            self.driver.get(url)
            sleep(5)
        except Exception as e:
           raise  e
       
    # Método para enviar teclas de forma dinámica a elementos
    def send_keys_dynamic(self, elements):
        try:
            for element in elements:
                xpath, value = element.get("relative"), element.get("value")
                input_element = self.get_element(by="xpath",value=xpath)
                if input_element:
                    input_element.clear()
                    input_element.send_keys(str(value))
                else:
                    raise ElementNotFoundException(
                        message="Element not found",
                        error=f"Element not found for xpath: {xpath}",
                        status_code=HTTPStatus.NOT_FOUND
                    )
        except Exception as e:
            raise e

    # Método para hacer clic en un elemento
    def click_element(self, element):
        try:
            relative = element.get("relative")
            click_element = self.get_element(by="xpath", value=relative)
            if click_element:
                click_element.click()
            else:
                raise ElementNotFoundException(
                    message="Element not found",
                    error=f"Element not found for xpath: {relative}",
                    status_code=HTTPStatus.NOT_FOUND
                )
        except Exception as e:
            raise e
       
    # Método para cerrar el navegador     
    def close_browser(self):
        try:
            self.driver.close()
        except Exception as e:
            raise e
            
    # Método para generar XPaths para elementos de una etiqueta específica
    def generate_xpaths_for_elements(self, tag_name_to_search):
        try:
            # Encontrar todos los elementos si tag_name_to_search es "all_tags"
            self.elements = self.driver.find_elements(By.TAG_NAME, tag_name_to_search)
            # Buscar elementos dentro de iframes
            iframe_elements = self.__find_elements_in_iframes(tag_name_to_search)
            # Generar XPaths para cada elemento encontrado
            xpaths_list = []

            for el in self.elements:
                result = self.__process_single_element(el, tag_name_to_search)
                xpaths_list.append(result)

            xpaths_list = xpaths_list + iframe_elements
            return xpaths_list

        except Exception as e:
            raise e
        
    def get_all_tags(self):
        all_tags = set()

        def recursive_tag_extraction(element):
            nonlocal all_tags

            try:
                self.driver.switch_to.frame(element)
                current_tags = self.__extract_tags(self.driver.page_source)
                all_tags.update(current_tags)

                for iframe_child in self.driver.find_elements(By.TAG_NAME, 'iframe'):
                    recursive_tag_extraction(iframe_child)

                self.driver.switch_to.parent_frame()
            except Exception as e:
                raise e

        # Start with the main page
        recursive_tag_extraction(None)
        
        # Omitir la etiqueta 'html' si está presente
        all_tags.discard('html')
        
        return list(all_tags)
    

    def __extract_tags(self,html):
        return {tag.name for tag in BeautifulSoup(html, 'html.parser').find_all()}
    
    
    def __find_elements_in_iframes(self, tag_name_to_search, iframe_path=""):
        xpaths_list = []

        # Obtener todos los iframes presentes en la página
        iframes = self.driver.find_elements(By.TAG_NAME, 'iframe')
        
        for iframe in iframes:

            # Cambiar al iframe
            path_iframe = self.driver.execute_script(script_path_absolute, iframe)

            # Concatenar el iframe_path actual con el path del iframe actual
            current_iframe_path = f"{iframe_path} > {path_iframe}" if iframe_path else path_iframe
        
            self.driver.switch_to.frame(iframe)
            
            # Llamada recursiva para manejar iframes anidados
            nested_iframe_elements = self.__find_elements_in_iframes(tag_name_to_search,current_iframe_path)
            xpaths_list.extend(nested_iframe_elements)
            
            self.elements_in_iframe = self.driver.find_elements(By.TAG_NAME, tag_name_to_search)
            # Buscar elementos dentro del iframe con el tag_name_to_search especificado
            
            # Agregar las etiquetas de los elementos encontrados a la lista
            for el in self.elements_in_iframe:
                result = self.__process_single_element(el, tag_name_to_search)
                result["iframe"] = True
                result["iframe_path"] = current_iframe_path
                xpaths_list.append(result)
                
            # Regresar al documento principal
            self.driver.switch_to.parent_frame()

        return xpaths_list


    def __process_single_element(self, element, tag_name_to_search):
        tag_name = element.tag_name
        id_attr = element.get_attribute('id')

        # Obtener el texto del elemento
        text_content = element.text.strip()       
        
        # Si el elemento no tiene texto, intentar obtener del atributo 'aria-label'
        if not text_content:
            text_content = element.get_attribute('aria-label')

        # Si aún no hay texto, intentar encontrar una etiqueta 'label' asociada por 'for' e 'id'
        if not text_content and id_attr:
            label_element = self.driver.execute_script(
                'return document.querySelector("label[for=\'{}\']")'.format(id_attr)
            )
            if label_element:
                text_content = label_element.text.strip()

        # Si todavía no hay texto, intentar obtener del atributo 'placeholder'
        if not text_content:
            text_content = element.get_attribute('placeholder')

        # Si todavía no hay texto, intentar obtener del atributo 'value'
        if not text_content:
            text_content = element.get_attribute('value')

        # Generar XPath absoluto
        xpath_absolute = self.driver.execute_script(script_path_absolute, element)

        # Generar XPath relativo
        parent = element.find_element(By.XPATH, "./..")
        children_same_tag = parent.find_elements(By.TAG_NAME, tag_name_to_search)

        # Verificar si el elemento tiene ID
        if id_attr:
            # Obtener el índice correcto para el XPath relativo
            index = 1
            for sibling in children_same_tag:
                if sibling == element:
                    break
                if sibling.get_attribute('id') == id_attr:
                    index += 1

            xpath_relative = f"//{tag_name_to_search}[@id='{id_attr}'][{index}]"
        else:
            # Si no tiene ID, usar contains con el texto del elemento
            xpath_relative = f"//{tag_name_to_search}[contains(text(), '{text_content}')]"
            # Verificar si el elemento está dentro de un iframe
        is_iframe = False
        if parent.tag_name.lower() == 'iframe':
            is_iframe = True
            
        return {"tag_name": tag_name, "text": text_content, "iframe": is_iframe, "absolute": xpath_absolute, "relative": xpath_relative}





