import re

class XPathProcessor:
    
    @staticmethod
    def validate_xpath_completion(instruction, elements, regex_pattern):
        """
        Valida si los campos especificados en la instrucción están presentes en los textos de los elementos.
        """
        instruction_fields = set(
            field for field, value in re.findall(regex_pattern, instruction)
        )

        elements_texts = set(item['text'] for item in elements if 'text' in item)

        are_filled = instruction_fields.issubset(elements_texts)
        return are_filled

    @staticmethod
    def filter_elements_by_xpath_and_values(instruction, elements, regex_pattern):
        """
        Filtra los elementos por XPath(absolute,relative) y valores especificados en la instrucción.
        """
        elements_xpaths = [
            {"absolute": element['absolute'], "relative": element.get('relative'), "value": element.get('text')}
            for element in elements if 'absolute' in element
        ]

        input_values = re.findall(regex_pattern, instruction)

        filtered_elements = [
            {
                "absolute": element['absolute'],
                "relative": element['relative'],
                "value": next(
                    (value for field, value in input_values if field.lower() in (element['value'] or '').lower()),
                    None
                )
            }
            for element in elements_xpaths if 'value' in element
        ]

        filtered_list = [d for d in filtered_elements if d['value'] is not None]

        return filtered_list

    @staticmethod
    def find_element_by_text(elements, target_label):
        """
        Encuentra el elemento cuyo texto coincide con la etiqueta objetivo.
        """
        matching_element = next(
            (element for element in elements if element.get('text', '').lower() == target_label.lower()), None
        )
        return matching_element
