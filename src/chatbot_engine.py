"""
chatbot_engine.py
-----------------
Motor conversacional simbólico y arquitectura base para clasificación de intenciones.
Define una interfaz abstracta (BaseIntentClassifier) para permitir desacoplamiento y
escalabilidad hacia modelos de Machine Learning (TF-IDF + SVM) en etapas futuras.
"""

from abc import ABC, abstractmethod
import json
import os
import random
from typing import Dict, List, Optional, Any, Set
from src.preprocessor import normalizar, tokenizar_y_stem


class ChatbotEngineError(Exception):
    """Excepción para errores de inicialización o procesamiento en el motor."""
    pass


class BaseIntentClassifier(ABC):
    """
    Contrato base (Strategy pattern) para los clasificadores de intención.
    Permite intercambiar el motor de reglas por un clasificador supervisado de ML
    en el Avance 2 sin alterar el gestor de diálogo.
    """

    @abstractmethod
    def clasificar(self, mensaje: str) -> str:
        """Determina la intención comunicativa a partir de la entrada del usuario."""
        pass

    @abstractmethod
    def obtener_intenciones(self) -> List[str]:
        """Retorna la lista de etiquetas de intención soportadas."""
        pass

    @abstractmethod
    def obtener_respuesta_base(self, intencion: str) -> str:
        """Retorna una respuesta canónica o predeterminada para la intención."""
        pass


class RuleBasedChatbot(BaseIntentClassifier):
    """
    Clasificador de intenciones basado en reglas simbólicas, NLTK stemming
    y coincidencia ponderada de palabras clave y frases compuestas.
    """

    def __init__(self, intents_filepath: str) -> None:
        self._filepath: str = intents_filepath
        self._intents_data: Dict[str, Any] = {}
        self._reglas: Dict[str, Dict[str, Any]] = {}
        self._respuestas: Dict[str, List[str]] = {}
        self._cargar_conocimiento()
        self._construir_indice_reglas()

    def _cargar_conocimiento(self) -> None:
        """Lee el catálogo de intenciones asegurando manejo seguro de archivos."""
        if not os.path.exists(self._filepath):
            raise ChatbotEngineError(
                f"[ERROR] Catálogo de intenciones no encontrado en '{self._filepath}'"
            )
        try:
            with open(self._filepath, mode="r", encoding="utf-8") as f:
                self._intents_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ChatbotEngineError(
                f"[ERROR] Formato JSON inválido en '{self._filepath}': {e}"
            ) from e

        if "intents" not in self._intents_data:
            raise ChatbotEngineError(
                "[ERROR] Estructura inválida: no se encontró la clave raíz 'intents'."
            )

    def _construir_indice_reglas(self) -> None:
        """
        Precomputa las raíces léxicas (stemming) de palabras clave y frases
        para optimizar la clasificación en tiempo de ejecución.
        """
        for intent in self._intents_data.get("intents", []):
            tag = intent.get("tag")
            if not tag or tag == "fallback":
                continue

            keywords = intent.get("keywords", [])
            frases = intent.get("frases_clave", [])
            self._respuestas[tag] = intent.get("responses", [])

            # Raíces de palabras clave individuales
            palabras_stem: Set[str] = {
                stem for kw in keywords for stem in tokenizar_y_stem(kw)
            }

            # Secuencias de raíces para frases clave compuestas
            frases_stem: List[List[str]] = [
                tokenizar_y_stem(frase) for frase in frases if frase.strip()
            ]

            self._reglas[tag] = {
                "palabras": palabras_stem,
                "frases": frases_stem
            }

        # Guardar respuestas para fallback
        for intent in self._intents_data.get("intents", []):
            if intent.get("tag") == "fallback":
                self._respuestas["fallback"] = intent.get("responses", [])

    @staticmethod
    def _contiene_secuencia(tokens: List[str], secuencia: List[str]) -> bool:
        """Verifica si una secuencia de stems aparece de forma contigua en los tokens."""
        n = len(secuencia)
        if n == 0 or n > len(tokens):
            return False
        return any(tokens[i : i + n] == secuencia for i in range(len(tokens) - n + 1))

    def clasificar(self, mensaje: str) -> str:
        """
        Calcula puntaje por coincidencia:
          - Palabra clave individual: 1 punto
          - Frase clave compuesta: 3 puntos
        Retorna la etiqueta con mayor puntaje o 'fallback'.
        """
        tokens_usuario = tokenizar_y_stem(mensaje)
        if not tokens_usuario:
            return "fallback"

        conjunto_usuario = set(tokens_usuario)
        mejor_intencion = "fallback"
        mejor_puntaje = 0

        for tag, regla in self._reglas.items():
            # Puntos por palabras individuales
            puntaje = len(conjunto_usuario & regla["palabras"])
            # Puntos adicionales por frases clave completas
            for frase_seq in regla["frases"]:
                if self._contiene_secuencia(tokens_usuario, frase_seq):
                    puntaje += 3

            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor_intencion = tag

        return mejor_intencion if mejor_puntaje > 0 else "fallback"

    def obtener_intenciones(self) -> List[str]:
        """Retorna las etiquetas (tags) registradas en el catálogo."""
        return [item["tag"] for item in self._intents_data.get("intents", [])]

    def obtener_respuesta_base(self, intencion: str) -> str:
        """Retorna una respuesta canónica para la intención."""
        respuestas = self._respuestas.get(intencion, [])
        if respuestas:
            return random.choice(respuestas)
        # Si la intención no existe, toma una respuesta al azar del fallback
        respuestas_fallback = self._respuestas.get("fallback", ["Operación no reconocida."])
        return random.choice(respuestas_fallback)
