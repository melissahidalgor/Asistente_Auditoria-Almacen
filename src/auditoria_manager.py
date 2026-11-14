"""
auditoria_manager.py
--------------------
Gestor de estado, ciclo de vida de la auditoría y reglas de negocio del almacén.
Evalúa conformidad de empaques según la política de calidad y mantiene la trazabilidad mediante bitácora estructurada.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List


class AuditoriaSession:
    """
    Controla el contexto, estado transaccional y memoria de la sesión
    de inspección en andenes de carga.
    """

    def __init__(self, responsable: str = "Inspector_Anden") -> None:
        self.responsable: str = responsable
        self.lote: Optional[str] = None
        self.anden: Optional[int] = None
        self.fecha_registro: Optional[str] = None
        self.dictamen: Optional[str] = None
        self.en_progreso: bool = False
        self.pendiente: Optional[str] = None  # Slot pendiente: "lote" o "anden"
        self.inspeccion_pendiente: Optional[str] = None  # Estatus reportado antes de tener lote/andén
        self.bitacora: List[Dict[str, Any]] = []

    @property
    def identificadores_completos(self) -> bool:
        """Verifica si tanto el lote como el andén han sido registrados."""
        return bool(self.lote and self.anden is not None)

    def iniciar_inspeccion(self, lote: Optional[str] = None, anden: Optional[int] = None) -> None:
        """Inicializa una auditoría activa con marca de tiempo ISO."""
        self.en_progreso = True
        self.fecha_registro = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.actualizar_identificadores(lote=lote, anden=anden)
        self.registrar_evento(
            mensaje="Inicio de auditoría",
            intencion="iniciar_auditoria",
            evento="inicio_auditoria",
            detalle=f"Lote: {self.lote}, Andén: {self.anden}"
        )

    def actualizar_identificadores(self, lote: Optional[str] = None, anden: Optional[int] = None) -> None:
        """Actualiza lote y andén validando formatos."""
        if lote:
            self.lote = str(lote).strip().upper()
        if anden is not None:
            try:
                self.anden = int(anden)
            except (ValueError, TypeError):
                pass

    def evaluar_empaque(self, es_danado: bool) -> str:
        """
        Aplica las reglas de negocio de recepción:
        - Dañado: Rechazo de lote y traslado a cuarentena.
        - Conforme: Aprobación y liberación para ingreso a almacén.
        """
        self.dictamen = "RECHAZADO_CUARENTENA" if es_danado else "APROBADO_LIBERADO"
        return self.dictamen

    def registrar_evento(
        self,
        mensaje: str,
        intencion: str,
        evento: Optional[str] = None,
        detalle: Optional[str] = None
    ) -> None:
        """Guarda un registro en la bitácora transaccional para trazabilidad."""
        registro = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "responsable": self.responsable,
            "mensaje": mensaje,
            "intencion": intencion,
            "evento": evento,
            "detalle": detalle,
            "lote": self.lote,
            "anden": self.anden,
            "dictamen": self.dictamen,
            "en_progreso": self.en_progreso
        }
        self.bitacora.append(registro)

    def cerrar_auditoria(self) -> None:
        """Cierra el ciclo de inspección actual limpiando variables de sesión."""
        self.lote = None
        self.anden = None
        self.fecha_registro = None
        self.dictamen = None
        self.en_progreso = False
        self.pendiente = None
        self.inspeccion_pendiente = None

    def reset(self) -> None:
        """Limpia todo el estado de la sesión, incluyendo la bitácora en memoria."""
        self.cerrar_auditoria()
        self.bitacora.clear()

    def guardar_bitacora(self, ruta_archivo: str = "data/bitacora_sesion.json") -> None:
        """Persiste los registros de auditoría en formato JSON estructurado."""
        ruta = Path(ruta_archivo)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(self.bitacora, f, ensure_ascii=False, indent=2)

    def obtener_resumen_reporte(self) -> Dict[str, Any]:
        """Retorna resumen para bitácora o persistencia externa."""
        return {
            "responsable": self.responsable,
            "fecha": self.fecha_registro,
            "lote": self.lote,
            "anden": self.anden,
            "dictamen": self.dictamen,
            "en_progreso": self.en_progreso,
            "total_eventos": len(self.bitacora)
        }
