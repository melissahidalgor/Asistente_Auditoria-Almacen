"""
dialogue_manager.py
-------------------
Gestor de diálogo y orquestador multimodal del Asistente de Auditoría de Almacén.
Coordina la interacción entre el clasificador de intenciones de texto (BaseIntentClassifier),
el clasificador visual de empaques (VisionPackageClassifier), el extractor de entidades
(extraer_entidades) y la sesión activa (AuditoriaSession).
Implementa llenado de datos (slot-filling), inspección visual por CNN y bitácora.
"""

from typing import Optional, Dict, Any
from src.chatbot_engine import BaseIntentClassifier
from src.auditoria_manager import AuditoriaSession
from src.entity_extractor import extraer_entidades, extraer_numero_aislado
from src.vision_classifier import VisionPackageClassifier, VisionClassifierError


class DialogueManager:
    """
    Orquestador del flujo conversacional y multimodal (texto + imagen)
    para la recepción y auditoría de mercancías en almacén.
    """

    def __init__(
        self,
        classifier: BaseIntentClassifier,
        session: Optional[AuditoriaSession] = None,
        ruta_bitacora: Optional[str] = "data/bitacora_sesion.json",
        vision_classifier: Optional[VisionPackageClassifier] = None
    ) -> None:
        self.classifier: BaseIntentClassifier = classifier
        self.session: AuditoriaSession = session or AuditoriaSession()
        self.ruta_bitacora: Optional[str] = ruta_bitacora
        self.vision_classifier: VisionPackageClassifier = (
            vision_classifier or VisionPackageClassifier(lazy_load=True)
        )
        self.ultima_intencion: str = "fallback"
        self.ultimo_resultado_visual: Optional[Dict[str, Any]] = None

    def responder(self, mensaje: str = "", ruta_imagen: Optional[str] = None) -> str:
        """
        Procesa el turno conversacional o multimodal del usuario:
          1. Extrae entidades operativas del texto (lote y andén).
          2. Si se recibe 'ruta_imagen' desde la UI o consola, ejecuta el clasificador
             visual CNN (EfficientNetB0 / ResNet50) e interpreta su resultado.
          3. Si es solo texto, clasifica la intención léxica y resuelve slot-filling.
          4. Canaliza hacia la regla de negocio correspondiente.
        """
        entidades = extraer_entidades(mensaje) if mensaje else {"lote": None, "anden": None}

        # Flujo Multimodal: Si la interfaz o consola envía una imagen para inspección visual
        if ruta_imagen:
            return self._manejar_inspeccion_visual(mensaje, ruta_imagen, entidades)

        intencion = self.classifier.clasificar(mensaje)

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
        # Si el clasificador asignó fallback pero el usuario proporcionó entidades operativas (lote o andén)
        elif intencion == "fallback" and (entidades["lote"] is not None or entidades["anden"] is not None):
            intencion = "iniciar_auditoria"

        # Enrutamiento según intención
        if intencion == "iniciar_auditoria":
            respuesta = self._manejar_iniciar_auditoria(mensaje, intencion, entidades)
        elif intencion in ("reportar_dano_empaque", "confirmar_empaque_intacto"):
            respuesta = self._manejar_resultado_inspeccion(mensaje, intencion, entidades)
        elif intencion == "despedida":
            respuesta = self._manejar_despedida(mensaje, intencion)
        else:
            self.session.registrar_evento(mensaje=mensaje, intencion=intencion)
            respuesta = self.classifier.obtener_respuesta_base(intencion)

        self.ultima_intencion = intencion
        return respuesta

    def _manejar_inspeccion_visual(
        self,
        mensaje: str,
        ruta_imagen: str,
        entidades: Dict[str, Any]
    ) -> str:
        """
        Ejecuta la clasificación de la fotografía con la CNN, actualiza entidades
        y comunica la interpretación del diagnóstico visual al inspector.
        """
        try:
            resultado_vis = self.vision_classifier.clasificar_imagen(ruta_imagen)
        except (FileNotFoundError, VisionClassifierError) as e:
            self.ultima_intencion = "fallback"
            self.session.registrar_evento(
                mensaje=mensaje,
                intencion="error_imagen",
                evento="error_carga_imagen",
                detalle=str(e)
            )
            return f"[Visión IA] No fue posible analizar la imagen: {e}"

        self.ultimo_resultado_visual = resultado_vis
        intencion_visual = resultado_vis["intencion_equivalente"]
        etiqueta = resultado_vis["etiqueta"]
        confianza = resultado_vis["confianza"]
        prefijo_visual = f"[Visión IA: {etiqueta} ({confianza:.1%} confianza)] "

        self.ultima_intencion = intencion_visual
        respuesta_base = self._manejar_resultado_inspeccion(
            mensaje=mensaje,
            intencion=intencion_visual,
            entidades=entidades,
            detalle_extra=f"Imagen='{ruta_imagen}' -> {etiqueta} ({confianza:.1%})"
        )
        return prefijo_visual + respuesta_base

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

        # Ambos identificadores disponibles
        self.session.pendiente = None
        self.session.en_progreso = True

        # Si el inspector ya había reportado el estado del empaque (por texto o imagen) antes de completar lote/andén,
        # cerramos el registro asociando dicho estatus en lugar de volver a pedir la inspección física.
        if self.session.inspeccion_pendiente in ("reportar_dano_empaque", "confirmar_empaque_intacto"):
            intencion_previa = self.session.inspeccion_pendiente
            es_dano = (intencion_previa == "reportar_dano_empaque")
            evento = "no_conformidad" if es_dano else "empaque_conforme"
            dictamen = self.session.evaluar_empaque(es_danado=es_dano)
            lote_actual = self.session.lote
            anden_actual = self.session.anden

            self.session.registrar_evento(
                mensaje=mensaje,
                intencion=intencion,
                evento=evento,
                detalle=f"Lote {lote_actual} en Andén {anden_actual} -> {dictamen}"
            )
            self.session.cerrar_auditoria()

            if es_dano:
                return (
                    f"[Lote {lote_actual} · andén {anden_actual}] Registro completado. "
                    f"Se ha asociado la alerta de daño reportada previamente (Dictamen: {dictamen}). "
                    "El lote queda rechazado en zona de cuarentena."
                )
            return (
                f"[Lote {lote_actual} · andén {anden_actual}] Registro completado. "
                f"Se ha asociado la confirmación de empaque intacto reportada previamente (Dictamen: {dictamen}). "
                "El material queda aprobado y liberado para estiba."
            )

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

    def _manejar_resultado_inspeccion(
        self,
        mensaje: str,
        intencion: str,
        entidades: Optional[Dict[str, Any]] = None,
        detalle_extra: Optional[str] = None
    ) -> str:
        """Procesa el reporte de daño o conformidad (textual o visual) y emite el dictamen de calidad."""
        # Capturar lote o andén si el inspector los incluyó en el mismo mensaje del reporte
        if entidades:
            if entidades.get("lote"):
                self.session.lote = entidades["lote"]
            if entidades.get("anden") is not None:
                self.session.anden = entidades["anden"]

        es_dano = (intencion == "reportar_dano_empaque")
        evento = "no_conformidad" if es_dano else "empaque_conforme"
        base_resp = self.classifier.obtener_respuesta_base(intencion)

        if self.session.identificadores_completos:
            self.session.en_progreso = True
            self.session.evaluar_empaque(es_danado=es_dano)
            dictamen_texto = f"[Lote {self.session.lote} · andén {self.session.anden}] {base_resp}"
            detalle_log = f"{self.session.dictamen} ({detalle_extra})" if detalle_extra else self.session.dictamen
            self.session.registrar_evento(
                mensaje=mensaje,
                intencion=intencion,
                evento=evento,
                detalle=detalle_log
            )
            # Cierra ciclo únicamente cuando el dictamen ya quedó asociado a lote y andén
            self.session.cerrar_auditoria()
            return dictamen_texto

        # Faltan identificadores: conservar el estatus reportado y solicitar el dato faltante
        self.session.inspeccion_pendiente = intencion
        if self.session.lote is None and self.session.anden is not None:
            self.session.pendiente = "lote"
            nota = f"He anotado el andén {self.session.anden}; por favor indícame el número de lote para asociar este registro."
        elif self.session.anden is None and self.session.lote is not None:
            self.session.pendiente = "anden"
            nota = f"He anotado el lote {self.session.lote}; por favor indícame el andén para asociar este registro."
        else:
            self.session.pendiente = "lote"
            nota = "Nota: no hay una auditoría activa; indícame el lote y el andén para asociar este registro."

        detalle_pend = f"PENDIENTE_IDENTIFICADORES ({detalle_extra})" if detalle_extra else "PENDIENTE_IDENTIFICADORES"
        self.session.registrar_evento(
            mensaje=mensaje,
            intencion=intencion,
            evento=evento,
            detalle=detalle_pend
        )
        return f"{base_resp} {nota}"

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
