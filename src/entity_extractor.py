    # ---------- Entidades (expresiones regulares) ----------
from src.preprocessor import normalizar
import re
from typing import Optional, TypedDict

class EntidadesAuditoria(TypedDict):
    lote: Optional[str]
    anden: Optional[str]

def extraer_entidades(mensaje: str) -> EntidadesAuditoria:
    """
    Extrae identificadores operativos (lote y andén/bahía/muelle).
    Soporta identificadores numéricos y alfanuméricos con guiones.
    """
    texto = normalizar(mensaje)
    
    # Captura: lote 504, lote no 12b, lote batch-88
    match_lote = re.search(
        r"\b(?:lote|batch|orden)\s+(?:numero\s+|no\s+)?([a-z0-9_-]+)", 
        texto
    )
    
    # Captura: anden 3, bahia b, muelle 4a
    match_anden = re.search(
        r"\b(?:anden|bahia|muelle|rampa)\s+(?:numero\s+|no\s+)?([a-z0-9]+)", 
        texto
    )
    
    return {
        "lote": match_lote.group(1).upper() if match_lote else None,
        "anden": match_anden.group(1).upper() if match_anden else None
    }