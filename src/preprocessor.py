"""
preprocessor.py
---------------
Módulo de preprocesamiento de lenguaje natural (NLP) con Python y NLTK.
Aplica las 4 técnicas estándar requeridas para procesamiento de texto:
  1. Normalización (minúsculas, remoción canónica NFD de acentos/diacríticos y caracteres).
  2. Tokenización léxica (wordpunct_tokenize).
  3. Eliminación de palabras vacías (stopwords en español con NLTK, protegiendo términos de dominio).
  4. Reducción morfológica mediante Stemming en español (SnowballStemmer).
"""

import re
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize

# Inicialización del stemmer para idioma español
_STEMMER = SnowballStemmer("spanish")


def normalizar(texto: str) -> str:
    """
    Normaliza el texto eliminando acentos/diacríticos, pasando a minúsculas
    y limpiando signos de puntuación y espacios redundantes.
    Conserva guiones y caracteres alfanuméricos útiles para lotes.
    """
    if not texto:
        return ""
    # Descomposición canónica NFD para aislar marcas diacríticas
    texto = unicodedata.normalize("NFD", texto.lower().strip())
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    # Sustituye caracteres no alfanuméricos por espacios (conservando guiones y números)
    texto = re.sub(r"[^a-z0-9\s_-]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


# Carga segura del corpus de stopwords de NLTK
try:
    _STOPWORDS_RAW: set[str] = set(stopwords.words("spanish"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOPWORDS_RAW = set(stopwords.words("spanish"))

# Exclusión de palabras críticas del dominio operativo del almacén:
# - 'sin', 'no', 'nada': determinan negación ("sin daños", "no conforme")
# - 'hasta': clave en despedidas ("hasta luego", "hasta mañana")
# - 'estado', 'estados': clave en evaluación física ("en buen estado", "perfecto estado")
# - 'contra': clave en política ("no conformidad")
PALABRAS_PROTEGIDAS: set[str] = {"sin", "no", "nada", "hasta", "estado", "estados", "contra"}

# Conjunto final de stopwords normalizadas usando la función normalizar()
STOPWORDS_ES: set[str] = {normalizar(p) for p in _STOPWORDS_RAW if normalizar(p)} - PALABRAS_PROTEGIDAS


def preprocesar(texto: str, eliminar_stopwords: bool = False) -> list[str]:
    """
    Aplica tokenización con NLTK, filtrado opcional de stopwords y reducción
    morfológica a la raíz mediante SnowballStemmer.
    
    :param texto: Cadena de texto de entrada.
    :param eliminar_stopwords: Si es True, filtra las palabras vacías de STOPWORDS_ES.
                               Por defecto es False para mantener compatibilidad con reglas.
    :return: Lista de stems resultantes.
    """
    texto_norm = normalizar(texto)
    if not texto_norm:
        return []
        
    tokens = [t for t in wordpunct_tokenize(texto_norm) if t.isalnum()]
    
    if eliminar_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_ES]
        
    return [_STEMMER.stem(t) for t in tokens]
