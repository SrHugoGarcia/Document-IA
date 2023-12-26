from abc import ABC, abstractmethod

class ScanServices(ABC):
    @abstractmethod
    def process_and_scan_elements(self, instruction: str):
        pass

    @abstractmethod
    def login_and_scan_elements(self, instruction: str):
        pass

    @abstractmethod
    def process_and_scan_elements_v2(self, url: str,tag: str,browser: str):
        pass