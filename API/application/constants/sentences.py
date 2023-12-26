sentences_scan_public = [
        "La aplicación web .* se inicia en .* en una nueva ventana.",
        "Se realiza un barrido de elementos identificado como .*."
]


elementos_dom = [
"a","abbr","address","area","article","aside","audio",
"b","bdi","bdo","blockquote","body","button",
"canvas","caption","cite","code","col","colgroup",
"data","datalist","dd","del","details","dfn","div","dl","dt",
"em",
"fieldset","figcaption","figure","footer",
"h1","h2","h3","h4","h5","head","header","hgroup","hr",
"i","iframe","img","input","ins",
"kbd","label","legend","li",
"main","map","mark","meta","meter",
"nav","noscript",
"object", "ol","optgroup","option","output",
"p","param","picture","pre","progress",
"q",
"rp","rt","ruby",
"s","samp","script","section","select","small","source","span","strong","style","sub","summary","sup",
"svg",
"table","tbody","td","textarea","tfoot","th","thead","time","title","tr","track",
"u","ul",
"var","video",
"wbr"
]#   "NOUN", "PROPN", "ADJ", "CCONJ", "VERB", "ADP", "INTJ", "ADV", "PRON"

browsers = [
    "chrome",
    "edge",
    "firefox"
]


login= [
    "La aplicación web .* se inicia en .* en una nueva ventana.",
    "El usuario espera a que aparezca un elemento designado como .*.",
    "La condición para avanzar es que el contenido del elemento coincida con la cadena .*.",
    "El usuario completa el formulario correspondiente donde (.*).",
    "El usuario hace clic en el .* designado con el texto .*.",
    "El usuario espera a que aparezca un elemento designado como .*.",
    "El usuario aguarda la aparición de un elemento específico identificado como .*.",
    "La condición para ejecutar un barrido de elementos es que el contenido del elemento se corresponda con la palabra (.*).",
    "Se procede a realizar un barrido de elementos de .*."
]

element_clicks = ["button", "link", "span", "a","input"]


#uuid
#IFRAME
#LOGIN
#BARRIDO
#NOTES