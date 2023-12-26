from enums import browser
from selenium import webdriver
from selenium.webdriver.remote.webdriver import WebDriver
from exceptions import BrowserNotFoundException
from http import HTTPStatus
from enums.browser import Browser
class SeleniumConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SeleniumConfig, cls).__new__(cls)
        return cls._instance

    @staticmethod
    def get_instance():
        if not SeleniumConfig._instance:
            SeleniumConfig._instance = SeleniumConfig()
        return SeleniumConfig._instance

    @staticmethod
    def get_browser_instance(browser_name: str) -> WebDriver:
        if browser_name == Browser.CHROME.value:
            return webdriver.Chrome()
        elif browser_name == Browser.FIREFOX.value:
            return webdriver.Firefox()
        elif browser_name == Browser.EDGE.value:
            return webdriver.Edge()
        else:
            raise BrowserNotFoundException(f"Unrecognized browser: {browser_name}","Invalid browser",HTTPStatus.BAD_REQUEST)