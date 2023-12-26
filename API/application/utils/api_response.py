from flask import jsonify
from http import HTTPStatus
import json

class APIResponse:
    @staticmethod
    def convert_to_json(keys, values):
        # Convierte dos listas en una cadena con formato JSON
        data_dict = dict(zip(keys, values))
        data_json = json.dumps(data_dict, indent=2)
        return json.loads(data_json)

    @staticmethod
    def success(data=None, message=None, status_code=HTTPStatus.OK):
        response_data = json.dumps({"status": "successful", "data": data, "message": message}, separators=(',', ':'))
        return response_data, status_code, {'Content-Type': 'application/json'}