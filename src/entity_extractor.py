"""
entity_extractor.py
-------------------
Extracción de entidades operativas (lote y andén/bahía/muelle)
mediante expresiones regulares y técnicas de normalización.
Soporta captura en un solo turno o por turnos separados (slot-filling).
"""

import re
from typing import Optional, TypedDict, List
from src.preprocessor import normalizar


class EntidadesAuditoria(TypedDict):
    lote: Optional[str]
    anden: Optional[int]


def extraer_entidades(mensaje: str) -> EntidadesAuditoria:
    """
    Extrae identificadores operativos:
      - lote: códigos numéricos o alfanuméricos precedidos por 'lote', 'batch', 'orden'.
      - anden: identificador numérico de andén, bahía, muelle o rampa.
    """
    texto = normalizar(mensaje)

    # Captura patrones como: "lote 504", "lote no. 12B", "batch 88", "lote batch-01"
    match_lote = re.search(
        r"\b(?:lote|batch|orden)\s+(?:numero\s+|no\s+)?([a-z0-9_-]+)",
        texto
    )

    # Captura patrones como: "anden 2", "bahia 3", "muelle 1", "rampa 4"
    match_anden = re.search(
        r"\b(?:anden|bahia|muelle|rampa)\s+(?:numero\s+|no\s+)?(\d+)",
        texto
    )

    lote_val: Optional[str] = match_lote.group(1).upper() if match_lote else None
    anden_val: Optional[int] = int(match_anden.group(1)) if match_anden else None

    return {
        "lote": lote_val,
        "anden": anden_val
    }


def extraer_numero_aislado(mensaje: str) -> Optional[int]:
    """
    Extrae un valor numérico único cuando el usuario responde brevemente
    a una solicitud de dato pendiente (ej. usuario escribe solo '3' o '710').
    """
    numeros = re.findall(r"\b\d+\b", mensaje.strip())
    if len(numeros) == 1:
        return int(numeros[0])
    return None
