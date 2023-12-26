from abc import ABC, abstractmethod

class SeleniumServices(ABC):
    @abstractmethod
    def get_element(self, by, value):
        pass

    @abstractmethod
    def wait_for_element(self, by, value):
        pass

    @abstractmethod
    def get_current_url(self):
        pass
    
    @abstractmethod
    def open_url(self,url):
        pass
    
    @abstractmethod       
    def send_keys_dynamic(self, elements):
        pass
    
    @abstractmethod    
    def click_element(self, element):
        pass
    
    @abstractmethod
    def close_browser(self):
        pass
    
    @abstractmethod
    def generate_xpaths_for_elements(self, tag_name_to_search):
        pass