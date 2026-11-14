"""
vision_classifier.py
--------------------
Clasificador visual de empaques basado en Redes Neuronales Convolucionales (CNN)
con Transferencia de Aprendizaje (EfficientNetB0 / ResNet50).
Permite inspeccionar fotografías de cajas en recepción de almacén y determinar
si el empaque se encuentra intacto ('caja_intacta') o dañado ('caja_danada').
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


class VisionClassifierError(Exception):
    """Excepción para errores de carga o inferencia en el módulo de visión."""
    pass


class VisionPackageClassifier:
    """
    Encapsula el modelo CNN entrenado en Keras para clasificar imágenes de empaques.
    Utiliza carga diferida (lazy loading) para optimizar el tiempo de arranque.
    """

    def __init__(
        self,
        model_filepath: str = "models/clasificador_cajas.keras",
        metadata_filepath: str = "models/metadata_vision.json",
        lazy_load: bool = True
    ) -> None:
        self._model_filepath: str = model_filepath
        self._metadata_filepath: str = metadata_filepath
        self._model: Any = None
        self._metadata: dict[str, Any] = {
            "modelo_ganador": "EfficientNetB0 (Transfer Learning)",
            "img_size": [224, 224],
            "etiquetas_espanol": ["caja_danada", "caja_intacta"],
            "umbral_binario": 0.5
        }
        self._cargar_metadata()
        if not lazy_load:
            self._asegurar_modelo()

    def _cargar_metadata(self) -> None:
        """Lee el archivo JSON de metadatos del modelo visual si existe."""
        if os.path.exists(self._metadata_filepath):
            try:
                with open(self._metadata_filepath, mode="r", encoding="utf-8") as f:
                    datos = json.load(f)
                    self._metadata.update(datos)
            except Exception as e:
                raise VisionClassifierError(
                    f"[ERROR] No se pudo leer '{self._metadata_filepath}': {e}"
                ) from e

    def _asegurar_modelo(self) -> None:
        """Carga el modelo Keras en memoria la primera vez que se solicita inferencia."""
        if self._model is not None:
            return

        if not os.path.exists(self._model_filepath):
            raise VisionClassifierError(
                f"[ERROR] Modelo visual no encontrado en '{self._model_filepath}'. "
                "Ejecute 'entrenamiento_vision.ipynb' y coloque el archivo en 'models/'."
            )
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
            import tensorflow as tf
            self._model = tf.keras.models.load_model(self._model_filepath)
        except Exception as e:
            raise VisionClassifierError(
                f"[ERROR] Fallo al cargar el modelo visual '{self._model_filepath}': {e}"
            ) from e

    @property
    def nombre_modelo(self) -> str:
        """Retorna el nombre de la arquitectura ganadora."""
        return str(self._metadata.get("modelo_ganador", "CNN Transfer Learning"))

    def clasificar_imagen(self, ruta_imagen: str) -> dict[str, Any]:
        """
        Analiza una imagen de empaque y retorna el diagnóstico estructurado:
          - etiqueta: 'caja_danada' o 'caja_intacta'
          - intencion_equivalente: 'reportar_dano_empaque' o 'confirmar_empaque_intacto'
          - es_danada: bool
          - confianza: float (0.0 a 1.0)
          - modelo: nombre del modelo CNN
          - ruta_imagen: ruta normalizada
        """
        ruta = Path(ruta_imagen.strip().strip("\"'"))
        if not ruta.exists() or not ruta.is_file():
            raise FileNotFoundError(
                f"No se encontró el archivo de imagen en la ruta: '{ruta}'"
            )

        self._asegurar_modelo()

        ancho, alto = self._metadata.get("img_size", [224, 224])
        try:
            with Image.open(ruta) as img:
                img_rgb = img.convert("RGB").resize((ancho, alto))
                tensor = np.array(img_rgb, dtype="float32")[None, ...]
        except Exception as e:
            raise VisionClassifierError(
                f"El archivo '{ruta}' no es una imagen válida o está corrupto: {e}"
            ) from e

        prob_clase_1 = float(self._model.predict(tensor, verbose=0)[0][0])
        umbral = float(self._metadata.get("umbral_binario", 0.5))

        etiquetas = self._metadata.get("etiquetas_espanol", ["caja_danada", "caja_intacta"])
        idx_pred = 1 if prob_clase_1 >= umbral else 0
        etiqueta = etiquetas[idx_pred]
        confianza = prob_clase_1 if idx_pred == 1 else (1.0 - prob_clase_1)
        es_danada = (etiqueta == "caja_danada")
        intencion_eq = "reportar_dano_empaque" if es_danada else "confirmar_empaque_intacto"

        return {
            "etiqueta": etiqueta,
            "intencion_equivalente": intencion_eq,
            "es_danada": es_danada,
            "confianza": confianza,
            "probabilidad_raw": prob_clase_1,
            "modelo": self.nombre_modelo,
            "ruta_imagen": str(ruta)
        }
