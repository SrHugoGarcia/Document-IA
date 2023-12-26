from config.app import App
from dotenv import load_dotenv
import os
import logging

if __name__ == '__main__':
    # Configuración del logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()  # Registros solo en la consola
        ]
    )
    load_dotenv()  
    app_instance = App(os.environ.get('PORT'))
    app_instance.custom_error_handlers()
    app_instance.running()
