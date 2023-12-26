from flask import Flask, jsonify
from routes.scan_routes import ScanRoutes
from exceptions import UrlNotFoundException, BrowserNotFoundException, InstructionNotFoundException, \
    InvalidSentenceException
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException, WebDriverException
from http import HTTPStatus


class App:
    def __init__(self, port):
        self.app = Flask(__name__)
        self._scan_routes = ScanRoutes()
        self.routes()
        self.port = port

    def running(self):
        self.app.run(port=self.port,debug=True)  # Pasar el puerto al método run

    def routes(self):
        self.app.register_blueprint(self._scan_routes.api_bp, url_prefix='/api/v1/scans')

    def custom_error_handlers(self):

        @self.app.errorhandler(404)
        def not_found_error(error):
            return jsonify({"error": "Not Found", "status": "fail"}), 404

        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({"error": "Internal Server Error", "status": "fail"}), 500

        @self.app.errorhandler(UrlNotFoundException)
        def url_not_found_error(error):
            return jsonify({"error": error.error, "message": error.message, "status": "fail"}), error.status_code.value

        @self.app.errorhandler(BrowserNotFoundException)
        def browser_not_found_error(error):
            return jsonify({"error": error.error, "message": error.message, "status": "fail"}), error.status_code.value

        @self.app.errorhandler(InstructionNotFoundException)
        def instruction_not_found_error(error):
            return jsonify({"error": error.error, "message": error.message, "status": "fail"}), error.status_code.value

        @self.app.errorhandler(InvalidSentenceException)
        def invalid_sentence_error(error):
            return jsonify({"error": error.error, "message": error.message, "status": "fail"}), error.status_code.value

        @self.app.errorhandler(NoSuchElementException)
        def handle_no_such_element_exception(error):
            response = {
                'message': 'Element not found on the page',
                'error': str(error),
                'status_code': HTTPStatus.NOT_FOUND
            }
            return jsonify(response), HTTPStatus.NOT_FOUND

        @self.app.errorhandler(TimeoutException)
        def handle_timeout_exception(error):
            response = {
                'message': 'Timeout while waiting for element to be visible',
                'error': str(error),
                'status_code': HTTPStatus.REQUEST_TIMEOUT
            }
            return jsonify(response), HTTPStatus.REQUEST_TIMEOUT

        @self.app.errorhandler(StaleElementReferenceException)
        def handle_stale_element_reference_exception(error):
            response = {
                'message': 'Stale element reference encountered',
                'error': str(error),
                'status_code': HTTPStatus.BAD_REQUEST
            }
            return jsonify(response), HTTPStatus.BAD_REQUEST

        @self.app.errorhandler(WebDriverException)
        def handle_web_driver_exception(error):
            response = {
                'message': 'WebDriverException occurred',
                'error': str(error),
                'status_code': HTTPStatus.INTERNAL_SERVER_ERROR
            }
            return jsonify(response), HTTPStatus.INTERNAL_SERVER_ERROR
    
                # Manejar PermissionError
        @self.app.errorhandler(PermissionError)
        def handle_permission_error(error):
            response = {
                'message': 'PermissionError occurred',
                'error': str(error),
                'status_code': HTTPStatus.INTERNAL_SERVER_ERROR
            }
            return jsonify(response), HTTPStatus.INTERNAL_SERVER_ERROR

        # Manejar IsADirectoryError
        @self.app.errorhandler(IsADirectoryError)
        def handle_is_a_directory_error(error):
            response = {
                'message': 'IsADirectoryError occurred',
                'error': str(error),
                'status_code': HTTPStatus.INTERNAL_SERVER_ERROR
            }
            return jsonify(response), HTTPStatus.INTERNAL_SERVER_ERROR

        # Manejar FileNotFoundError
        @self.app.errorhandler(FileNotFoundError)
        def handle_file_not_found_error(error):
            response = {
                'message': 'FileNotFoundError occurred',
                'error': str(error),
                'status_code': HTTPStatus.INTERNAL_SERVER_ERROR
            }
            return jsonify(response), HTTPStatus.INTERNAL_SERVER_ERROR

        # Manejar OSError
        @self.app.errorhandler(OSError)
        def handle_os_error(error):
            response = {
                'message': 'OSError occurred',
                'error': str(error),
                'status_code': HTTPStatus.INTERNAL_SERVER_ERROR
            }
            return jsonify(response), HTTPStatus.INTERNAL_SERVER_ERROR
        
#iframea anidados 