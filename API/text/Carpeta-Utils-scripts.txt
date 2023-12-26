script_path_absolute = """
// Función para generar un XPath absoluto desde un elemento dado
function absoluteXPath(element) {
    var comp, comps = [];
    var parent = null;
    var xpath = '';
    var getPos = function(element) {
        var position = 1, curNode;
        if (element.nodeType == Node.ATTRIBUTE_NODE) {
            return null;
        }
        for (curNode = element.previousSibling; curNode; curNode = curNode.previousSibling) {
            if (curNode.nodeName == element.nodeName) {
                ++position;
            }
        }
        return position;
    };

    // Comienza la generación del XPath absoluto
    if (element instanceof Document) {
        return '/';  // Si el elemento es un documento, el XPath es simplemente '/'
    }
    
    // Itera desde el elemento hasta el ancestro más alto (Document)
    for (; element && !(element instanceof Document); element = element.nodeType == Node.ATTRIBUTE_NODE ? element.ownerElement : element.parentNode) {
        comp = comps[comps.length] = {};
        switch (element.nodeType) {
            case Node.TEXT_NODE:
                comp.name = 'text()';  // Nodo de texto, nombre 'text()'
                break;
            case Node.ATTRIBUTE_NODE:
                comp.name = '@' + element.nodeName;  // Nodo de atributo, nombre '@nombreDelAtributo'
                break;
            case Node.PROCESSING_INSTRUCTION_NODE:
                comp.name = 'processing-instruction()';  // Nodo de instrucción de procesamiento, nombre 'processing-instruction()'
                break;
            case Node.COMMENT_NODE:
                comp.name = 'comment()';  // Nodo de comentario, nombre 'comment()'
                break;
            case Node.ELEMENT_NODE:
                comp.name = element.nodeName;  // Nodo de elemento, nombre del elemento
                break;
        }
        comp.position = getPos(element);  // Obtiene la posición del elemento
    }

    // Construye el XPath concatenando los componentes
    for (var i = comps.length - 1; i >= 0; i--) {
        comp = comps[i];
        xpath += '/' + comp.name.toLowerCase();  // Agrega cada componente al XPath en minúsculas
        if (comp.position !== null) {
            xpath += '[' + comp.position + ']';  // Si hay posición, agrega la posición al XPath
        }
    }

    return xpath;  // Devuelve el XPath absoluto generado
}

// Llama a la función absoluteXPath con el elemento proporcionado como argumento
return absoluteXPath(arguments[0]);
"""
