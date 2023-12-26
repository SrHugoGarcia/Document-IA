# app/routes/excel_routes.py
from flask import Blueprint
from controllers.scan_controller import ScanController  # Corregir la importación

class ScanRoutes:
    def __init__(self):
        self.api_bp = Blueprint('scans', __name__)
        self._scan_controller = ScanController()
        self.setup_routes()

    def setup_routes(self):
        self.api_bp.add_url_rule('/public', methods=['POST'], view_func=self._scan_controller.scan_elements_url_public)
        self.api_bp.add_url_rule('/login', methods=['POST'], view_func=self._scan_controller.login_to_url_private)
        self.api_bp.add_url_rule('/', methods=['POST'], view_func=self._scan_controller.scan_elements_url)
