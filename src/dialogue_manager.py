"""
dialogue_manager.py
-------------------
Gestor de diálogo y orquestador conversacional del Asistente de Auditoría de Almacén.
Coordina la interacción entre el clasificador de intenciones (BaseIntentClassifier),
el extractor de entidades (extraer_entidades) y la sesión activa (AuditoriaSession).
Implementa llenado de datos (slot-filling), validación de reglas operativas y bitácora.
"""

from typing import Optional, Dict, Any
from src.chatbot_engine import BaseIntentClassifier
from src.auditoria_manager import AuditoriaSession
from src.entity_extractor import extraer_entidades, extraer_numero_aislado


class DialogueManager:
    """
    Orquestador del flujo conversacional para la recepción y auditoría de mercancías.
    """

    def __init__(
        self,
        classifier: BaseIntentClassifier,
        session: Optional[AuditoriaSession] = None,
        ruta_bitacora: Optional[str] = "data/bitacora_sesion.json"
    ) -> None:
        self.classifier: BaseIntentClassifier = classifier
        self.session: AuditoriaSession = session or AuditoriaSession()
        self.ruta_bitacora: Optional[str] = ruta_bitacora
        self.ultima_intencion: str = "fallback"

    def responder(self, mensaje: str) -> str:
        """
        Procesa el turno conversacional del usuario:
          1. Clasifica la intención léxica.
          2. Extrae entidades operativas (lote y andén).
          3. Resuelve llenado de slots pendientes (slot-filling) si el usuario
             responde con un dato específico (ej. solo un número o identificador).
          4. Canaliza hacia la regla de negocio correspondiente.
        """
        intencion = self.classifier.clasificar(mensaje)
        entidades = extraer_entidades(mensaje)

        # Llenado contextual de slots pendientes (slot-filling)
        if self.session.pendiente:
            # Solo si no se detectó ninguna entidad explícita se intenta asignar número aislado
            if entidades["lote"] is None and entidades["anden"] is None:
                numero_aislado = extraer_numero_aislado(mensaje)
                if numero_aislado is not None:
                    intencion = "iniciar_auditoria"
                    if self.session.pendiente == "lote":
                        entidades["lote"] = str(numero_aislado)
                    elif self.session.pendiente == "anden":
                        entidades["anden"] = numero_aislado
            elif entidades["lote"] or entidades["anden"] is not None:
                intencion = "iniciar_auditoria"

        # Enrutamiento según intención
        if intencion == "iniciar_auditoria":
            respuesta = self._manejar_iniciar_auditoria(mensaje, intencion, entidades)
        elif intencion in ("reportar_dano_empaque", "confirmar_empaque_intacto"):
            respuesta = self._manejar_resultado_inspeccion(mensaje, intencion)
        elif intencion == "despedida":
            respuesta = self._manejar_despedida(mensaje, intencion)
        else:
            self.session.registrar_evento(mensaje=mensaje, intencion=intencion)
            respuesta = self.classifier.obtener_respuesta_base(intencion)

        self.ultima_intencion = intencion
        return respuesta

    def _manejar_iniciar_auditoria(
        self,
        mensaje: str,
        intencion: str,
        entidades: Dict[str, Any]
    ) -> str:
        """Valida identificadores operativos y controla el flujo de inicio de auditoría."""
        # Actualización de datos recolectados
        if entidades["lote"]:
            self.session.lote = entidades["lote"]
        if entidades["anden"] is not None:
            self.session.anden = entidades["anden"]

        # Verificación de completitud de slots
        if self.session.lote is None:
            self.session.pendiente = "lote"
            self.session.registrar_evento(
                mensaje=mensaje,
                intencion=intencion,
                evento="falta_lote"
            )
            if self.session.anden is not None:
                return f"Andén {self.session.anden} anotado. ¿Cuál es el número de lote que vas a auditar?"
            return "Entendido. ¿Cuál es el número de lote que vas a auditar?"

        if self.session.anden is None:
            self.session.pendiente = "anden"
            self.session.registrar_evento(
                mensaje=mensaje,
                intencion=intencion,
                evento="falta_anden"
            )
            return f"Lote {self.session.lote} anotado. ¿En qué andén o bahía se está recibiendo?"

        # Ambos identificadores disponibles: auditoría confirmada
        self.session.pendiente = None
        self.session.en_progreso = True
        self.session.registrar_evento(
            mensaje=mensaje,
            intencion=intencion,
            evento="inicio_auditoria",
            detalle=f"Lote {self.session.lote} en Andén {self.session.anden}"
        )

        # Respuesta estructurada guiando al inspector
        return (
            f"Entendido. He registrado el inicio de auditoría del lote {self.session.lote} "
            f"en el andén {self.session.anden}. Por favor, realiza la inspección física "
            "del paquete y describe si el empaque presenta golpes o si se encuentra intacto."
        )

    def _manejar_resultado_inspeccion(self, mensaje: str, intencion: str) -> str:
        """Procesa el reporte de daño o conformidad y emite el dictamen de calidad."""
        es_dano = (intencion == "reportar_dano_empaque")
        evento = "no_conformidad" if es_dano else "empaque_conforme"
        base_resp = self.classifier.obtener_respuesta_base(intencion)

        if self.session.en_progreso and self.session.identificadores_completos:
            dictamen = self.session.evaluar_empaque(es_danado=es_dano)
            dictamen_texto = f"[Lote {self.session.lote} · andén {self.session.anden}] {base_resp}"
        else:
            dictamen = "SIN_AUDITORIA_ACTIVA"
            dictamen_texto = (
                f"{base_resp} Nota: no hay una auditoría activa; indícame el lote y el andén "
                "para asociar este registro."
            )

        self.session.registrar_evento(
            mensaje=mensaje,
            intencion=intencion,
            evento=evento,
            detalle=self.session.dictamen
        )

        # Cierra ciclo tras registrar el dictamen
        self.session.cerrar_auditoria()
        return dictamen_texto

    def _manejar_despedida(self, mensaje: str, intencion: str) -> str:
        """Cierra la sesión y persiste la bitácora transaccional."""
        aviso = ""
        if self.session.en_progreso:
            aviso = f" Nota: la auditoría del lote {self.session.lote} quedó abierta sin resultado."

        self.session.registrar_evento(
            mensaje=mensaje,
            intencion=intencion,
            evento="fin_sesion"
        )

        if self.ruta_bitacora:
            self.session.guardar_bitacora(self.ruta_bitacora)

        respuesta = self.classifier.obtener_respuesta_base("despedida") + aviso
        self.session.cerrar_auditoria()
        return respuesta

