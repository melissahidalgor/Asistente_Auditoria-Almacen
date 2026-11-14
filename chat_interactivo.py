"""
chat_interactivo.py
-------------------
Punto de entrada interactivo multimodal para el Asistente de Auditoría de Almacén.
Permite a un inspector de andén interactuar con el chatbot desde la consola
combinando procesamiento de lenguaje natural (BoW + LinearSVC / Reglas) e
inspección visual de empaques mediante CNN (EfficientNetB0 Transfer Learning).
"""

import argparse
import re
import sys
from typing import Optional
from src.chatbot_engine import MLIntentClassifier, RuleBasedChatbot, BaseIntentClassifier
from src.auditoria_manager import AuditoriaSession
from src.dialogue_manager import DialogueManager
from src.vision_classifier import VisionPackageClassifier


def separar_mensaje_e_imagen(entrada: str) -> tuple[str, Optional[str]]:
    """
    Adaptador exclusivo de consola para simular el envío de imagen desde una UI.
    Detecta la palabra clave 'IMAGEN <ruta>' y separa el texto de la ruta del archivo.
    Ejemplos:
      - 'IMAGEN data/dataset_img/test/damagedpackages/foto.jpg'
      - 'lote 504 en anden 2 IMAGEN data/dataset_img/test/damagedpackages/foto.jpg'
    """
    match = re.search(r"\bIMAGEN\s+(.+)$", entrada, flags=re.IGNORECASE)
    if match:
        mensaje_texto = entrada[:match.start()].strip()
        ruta_imagen = match.group(1).strip().strip("\"'")
        return mensaje_texto, ruta_imagen
    return entrada.strip(), None


def crear_clasificador(modo: str) -> BaseIntentClassifier:
    """Instancia el clasificador de intenciones según el modo seleccionado."""
    if modo == "ml":
        try:
            return MLIntentClassifier(
                model_filepath="models/pipeline_intenciones.joblib",
                intents_filepath="data/intents.json"
            )
        except Exception as e:
            print(f"[ADVERTENCIA] No se pudo cargar el modelo de ML ({e}).")
            print("              Cambiando automáticamente al motor de reglas...")
            return RuleBasedChatbot("data/intents.json")
    else:
        return RuleBasedChatbot("data/intents.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Asistente Conversacional Multimodal para Auditoría de Almacén (Avance 2)"
    )
    parser.add_argument(
        "--modo",
        choices=["ml", "reglas"],
        default="ml",
        help="Motor de clasificación de intenciones de texto: 'ml' (por defecto) o 'reglas'."
    )
    args = parser.parse_args()

    # Configurar encoding para terminales Windows
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    clasificador_texto = crear_clasificador(args.modo)
    clasificador_vision = VisionPackageClassifier(lazy_load=True)

    nombre_motor_texto = (
        "Machine Learning (BoW + LinearSVC / SVM)"
        if isinstance(clasificador_texto, MLIntentClassifier)
        else "Reglas Simbólicas (NLTK Stemming - Avance 1)"
    )

    print("=" * 78)
    print(" 📦 ASISTENTE MULTIMODAL DE AUDITORÍA DE ALMACÉN - RECEPCIÓN DE MERCANCÍAS")
    print(f" 🧠 Motor Conversacional (NLP) : {nombre_motor_texto}")
    print(f" 📷 Motor de Visión (CNN)      : {clasificador_vision.nombre_modelo}")
    print(" 💡 Tip: Para adjuntar una foto escribe: IMAGEN <ruta_archivo>")
    print(" 📋 Escribe 'salir' para concluir la sesión y guardar la bitácora.")
    print("=" * 78)

    sesion = AuditoriaSession(responsable="Inspector_Consola")
    asistente = DialogueManager(
        classifier=clasificador_texto,
        session=sesion,
        ruta_bitacora="data/bitacora_sesion.json",
        vision_classifier=clasificador_vision
    )

    while True:
        try:
            entrada = input("\nInspector: ").strip()
            if not entrada:
                continue

            if entrada.lower() in ("salir", "exit", "quit"):
                respuesta = asistente.responder("Muchas gracias, hasta luego")
                print(f"Bot      : {respuesta}")
                print("\n[INFO] Sesión finalizada. Bitácora guardada en 'data/bitacora_sesion.json'.")
                break

            mensaje_texto, ruta_imagen = separar_mensaje_e_imagen(entrada)
            respuesta = asistente.responder(mensaje_texto, ruta_imagen=ruta_imagen)
            print(f"Bot      : {respuesta}")

        except (KeyboardInterrupt, EOFError):
            print("\n\n[INFO] Interrupción recibida. Cerrando sesión de auditoría...")
            asistente.responder("despedida")
            break


if __name__ == "__main__":
    main()
