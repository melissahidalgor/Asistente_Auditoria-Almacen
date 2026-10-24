"""
chat_interactivo.py
-------
Punto de entrada interactivo para el Asistente de Auditoría de Almacén.
Permite a un inspector de andén interactuar con el chatbot desde la consola.
"""

from src.chatbot_engine import RuleBasedChatbot
from src.auditoria_manager import AuditoriaSession
from src.dialogue_manager import DialogueManager


def main() -> None:
    print("=" * 70)
    print(" ASISTENTE DE AUDITORÍA DE ALMACÉN - RECEPCIÓN DE MERCANCÍAS")
    print(" Sistema conversacional para inspección y trazabilidad de empaques")
    print(" Escribe 'salir' para concluir la sesión y guardar la bitácora.")
    print("=" * 70)

    chatbot = RuleBasedChatbot("data/intents.json")
    sesion = AuditoriaSession(responsable="Inspector_Consola")
    asistente = DialogueManager(classifier=chatbot, session=sesion, ruta_bitacora="data/bitacora_sesion.json")

    while True:
        try:
            mensaje = input("\nInspector: ").strip()
            if not mensaje:
                continue

            if mensaje.lower() in ("salir", "exit", "quit"):
                respuesta = asistente.responder("Muchas gracias, hasta luego")
                print(f"Bot      : {respuesta}")
                print("\n[INFO] Sesión finalizada. Bitácora guardada en 'data/bitacora_sesion.json'.")
                break

            respuesta = asistente.responder(mensaje)
            print(f"Bot      : {respuesta}")

        except (KeyboardInterrupt, EOFError):
            print("\n\n[INFO] Interrupción recibida. Cerrando sesión de auditoría...")
            asistente.responder("despedida")
            break


if __name__ == "__main__":
    main()

