"""
preprocessor.py
---------------
Módulo de preprocesamiento de lenguaje natural (NLP) con Python y NLTK.
Aplica técnicas de limpieza estándar:
  1. Conversión a minúsculas y eliminación de acentos/diacríticos.
  2. Remoción de caracteres especiales y puntuación.
  3. Tokenización léxica (wordpunct_tokenize).
  4. Reducción morfológica mediante Stemming en español (SnowballStemmer).
"""

import unicodedata
import re
from typing import List
from nltk.stem import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize

# Inicialización del stemmer para idioma español (basado en reglas algorítmicas de Snowball)
_STEMMER = SnowballStemmer("spanish")


def normalizar(texto: str) -> str:
    """
    Normaliza el texto eliminando acentos/diacríticos, pasando a minúsculas
    y limpiando signos de puntuación y espacios redundantes.
    """
    if not texto:
        return ""
    # Descomposición canónica NFD para aislar marcas diacríticas
    texto = unicodedata.normalize("NFD", texto.lower().strip())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    # Sustituye caracteres no alfanuméricos por espacios
    texto = re.sub(r"[^a-z0-9\s_-]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()

def tokenizar_y_stem(texto: str) -> List[str]:
    """
    Aplica tokenización con NLTK y reduce cada token a su raíz (stem).
    Filtra tokens no alfanuméricos.
    """
    texto_norm = normalizar(texto)
    if not texto_norm:
        return []
    tokens = [t for t in wordpunct_tokenize(texto_norm) if t.isalnum()]
    return [_STEMMER.stem(t) for t in tokens]
