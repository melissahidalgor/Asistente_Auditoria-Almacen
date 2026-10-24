import unicodedata
import re

def normalizar(texto: str) -> str:
    """Minúsculas, remoción de acentos/diacríticos y normalización de espacios."""
    if not texto:
        return ""
    texto = unicodedata.normalize("NFD", texto.lower().strip())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    # Sustituye caracteres especiales por espacios y colapsa espacios en blanco repetidos
    texto = re.sub(r"[^\w\s]", " ", texto)
    return " ".join(texto.split())