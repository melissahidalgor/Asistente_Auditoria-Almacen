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
import joblib
from src.preprocessor import normalizar, preprocesar


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
                stem for kw in keywords for stem in preprocesar(kw)
            }

            # Secuencias de raíces para frases clave compuestas
            frases_stem: List[List[str]] = [
                preprocesar(frase) for frase in frases if frase.strip()
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
        tokens_usuario = preprocesar(mensaje)
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


class MLIntentClassifier(BaseIntentClassifier):
    """
    Clasificador de intenciones supervisado basado en Machine Learning (scikit-learn).
    Utiliza el pipeline ganador (CountVectorizer BoW + LinearSVC) entrenado con
    validación cruzada estratificada.
    """

    def __init__(
        self,
        model_filepath: str = "models/pipeline_intenciones.joblib",
        intents_filepath: str = "data/intents.json",
        umbral_decision: Optional[float] = None
    ) -> None:
        self._model_filepath: str = model_filepath
        self._intents_filepath: str = intents_filepath
        self._umbral_decision: Optional[float] = umbral_decision
        self._pipeline: Any = None
        self._respuestas: Dict[str, List[str]] = {}
        self._cargar_recursos()

    def _cargar_recursos(self) -> None:
        """Carga el modelo serializado con joblib y el catálogo de respuestas."""
        if not os.path.exists(self._model_filepath):
            raise ChatbotEngineError(
                f"[ERROR] Modelo entrenado no encontrado en '{self._model_filepath}'. "
                "Ejecute 'entrenamiento_ml.ipynb' para generar el artefacto."
            )
        try:
            self._pipeline = joblib.load(self._model_filepath)
        except Exception as e:
            raise ChatbotEngineError(
                f"[ERROR] No se pudo deserializar el pipeline en '{self._model_filepath}': {e}"
            ) from e

        if not os.path.exists(self._intents_filepath):
            raise ChatbotEngineError(
                f"[ERROR] Catálogo de intenciones no encontrado en '{self._intents_filepath}'"
            )
        try:
            with open(self._intents_filepath, mode="r", encoding="utf-8") as f:
                datos = json.load(f)
        except json.JSONDecodeError as e:
            raise ChatbotEngineError(
                f"[ERROR] Formato JSON inválido en '{self._intents_filepath}': {e}"
            ) from e

        for intent in datos.get("intents", []):
            tag = intent.get("tag")
            if tag:
                self._respuestas[tag] = intent.get("responses", [])

    def clasificar(self, mensaje: str) -> str:
        """
        Predice la intención del usuario a partir del texto ingresado.
        Si el texto está vacío o la certeza del hiperplano no supera el umbral,
        retorna 'fallback'.
        """
        if not mensaje or not mensaje.strip():
            return "fallback"

        # Si se configuró un umbral de margen para LinearSVC
        if self._umbral_decision is not None and hasattr(self._pipeline, "decision_function"):
            scores = self._pipeline.decision_function([mensaje])[0]
            max_score = float(max(scores))
            if max_score < self._umbral_decision:
                return "fallback"

        return str(self._pipeline.predict([mensaje])[0])

    def obtener_intenciones(self) -> List[str]:
        """Retorna las clases registradas en el estimador de scikit-learn."""
        if hasattr(self._pipeline, "classes_"):
            return list(self._pipeline.classes_)
        return list(self._respuestas.keys())

    def obtener_respuesta_base(self, intencion: str) -> str:
        """Retorna una respuesta canónica al azar para la intención dada."""
        respuestas = self._respuestas.get(intencion, [])
        if respuestas:
            return random.choice(respuestas)
        fallback = self._respuestas.get("fallback", ["Operación no reconocida."])
        return random.choice(fallback)

