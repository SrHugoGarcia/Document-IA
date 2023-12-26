from flask import request
from http import HTTPStatus

from utils.api_response import APIResponse
from services.scan_services_implements import ScanServicesImplements

class ScanController:
    def __init__(self):
        self._api_response = APIResponse()
        self._scan_services = ScanServicesImplements()

    def scan_elements_url_public(self):
        instruction = request.get_json().get('instruction')
        scan = self._scan_services.process_and_scan_elements(instruction)
        return self._api_response.success(scan, message="Successfully generated sweep", status_code=HTTPStatus.OK)

    def login_to_url_private(self):
        instruction = request.get_json().get('instruction')
        scan = self._scan_services.login_and_scan_elements(instruction)
        return self._api_response.success(scan, message="Successfully generated sweep", status_code=HTTPStatus.OK)
    
    def scan_elements_url(self):
        url = request.get_json().get('url')
        tag = request.get_json().get('tag')
        browser = request.get_json().get('browser')
        scan = self._scan_services.process_and_scan_elements_v2(url=url,tag=tag,browser=browser)
        return self._api_response.success(scan, message="Successfully generated sweep", status_code=HTTPStatus.OK)