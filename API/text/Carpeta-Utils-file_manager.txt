import os

class FileManager:
    def __init__(self, directory=None):
        self.directory = directory or '.'

    def write_content(self, content):
        try:
            with open(self.directory, 'w') as file:
                if isinstance(content, list):
                    content = ',\n'.join(map(str, content))
                file.write(content)
        except Exception as e:
            raise e

