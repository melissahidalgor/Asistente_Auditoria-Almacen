"""
run_tests.py
------------
Ejecutor automatizado de la batería de pruebas (data/test_conversaciones.json)
para validar el Asistente de Auditoría de Almacén.
Calcula métricas de precisión, tasa de fallbacks y genera reportes consolidados
en formato DataFrame de pandas, además de exportar la bitácora transaccional.
"""

import sys
import os
import json
from pathlib import Path
from typing import Optional
import pandas as pd

# Configurar encoding seguro para terminales Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Añadir raíz del proyecto al sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chatbot_engine import RuleBasedChatbot, MLIntentClassifier, BaseIntentClassifier
from src.auditoria_manager import AuditoriaSession
from src.dialogue_manager import DialogueManager


def cargar_pruebas_json(ruta_archivo: str = "data/test_conversaciones.json") -> dict:
    """Carga el dataset de pruebas estructurado en formato JSON."""
    ruta = Path(ruta_archivo)
    if not ruta.exists():
        raise FileNotFoundError(f"[ERROR] No se encontró el archivo de pruebas en: {ruta_archivo}")
    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def ejecutar_evaluacion(
    ruta_intents: str = "data/intents.json",
    ruta_test_json: str = "data/test_conversaciones.json",
    ruta_bitacora: str = "data/bitacora_pruebas.json",
    verbose: bool = True,
    clasificador: Optional[BaseIntentClassifier] = None,
    nombre_motor: str = "Reglas Simbólicas (Avance 1)"
) -> dict:
    """
    Ejecuta las conversaciones oficiales y pruebas de límites cargadas desde el JSON.
    Genera reportes de métricas en DataFrames de pandas y exporta la bitácora.
    """
    datos_pruebas = cargar_pruebas_json(ruta_test_json)
    conversaciones = datos_pruebas.get("conversaciones", [])
    pruebas_limite = datos_pruebas.get("pruebas_limite", [])

    chatbot = clasificador or RuleBasedChatbot(ruta_intents)
    bitacora_total = []

    total_turnos = 0
    aciertos_totales = 0
    fallbacks_totales = 0
    convs_exitosas = 0

    registro_conversaciones = []

    if verbose:
        print("\n" + "=" * 80)
        print(" BATERÍA DE PRUEBAS FUNCIONALES - ASISTENTE DE AUDITORÍA DE ALMACÉN")
        print(f" Motor Evaluado: {nombre_motor}")
        print(" Dataset de evaluación cargado desde:", ruta_test_json)
        print("=" * 80)

    for item in conversaciones:
        conv_id = item.get("id", "")
        nombre_conv = item.get("nombre", f"Conversación {conv_id}")
        turnos = item.get("turnos", [])

        if verbose:
            print(f"\n--- {nombre_conv} ---")

        # Cada conversación inicia con una sesión limpia
        session = AuditoriaSession(responsable="Inspector_Pruebas")
        dialogue = DialogueManager(classifier=chatbot, session=session, ruta_bitacora=None)
        
        turnos_conv = len(turnos)
        aciertos_conv = 0

        for t in turnos:
            mensaje = t.get("usuario", "")
            intencion_esperada = t.get("intencion_esperada", "")
            total_turnos += 1

            respuesta = dialogue.responder(mensaje)
            intencion_detectada = dialogue.ultima_intencion

            es_acierto = (intencion_detectada == intencion_esperada)
            if es_acierto:
                aciertos_totales += 1
                aciertos_conv += 1

            if intencion_detectada == "fallback":
                fallbacks_totales += 1

            if verbose:
                simbolo = "[OK]" if es_acierto else "[FALLO]"
                print(f"Usuario : {mensaje}")
                print(f"Bot     : {respuesta}")
                print(f"          [Esperada: {intencion_esperada:26} | Detectada: {intencion_detectada:26}] {simbolo}")

        es_perfecta = (aciertos_conv == turnos_conv)
        if es_perfecta:
            convs_exitosas += 1

        registro_conversaciones.append({
            "ID": conv_id,
            "Conversación": nombre_conv,
            "Turnos": turnos_conv,
            "Aciertos": aciertos_conv,
            "Precisión": f"{(aciertos_conv / turnos_conv):.1%}" if turnos_conv > 0 else "0.0%",
            "Estatus": "APROBADA" if es_perfecta else "FALLIDA"
        })

        # Acumular registros de bitácora
        bitacora_total.extend(session.bitacora)

    # Cálculo de métricas globales
    precision_global = aciertos_totales / total_turnos if total_turnos > 0 else 0.0
    tasa_fallback = fallbacks_totales / total_turnos if total_turnos > 0 else 0.0

    # 1. DataFrame Detalle por Conversación
    df_detalle = pd.DataFrame(registro_conversaciones)

    # 2. DataFrame Consolidado de Métricas
    df_metricas = pd.DataFrame([
        {"Métrica": "Conversaciones evaluadas", "Valor": str(len(conversaciones))},
        {"Métrica": "Conversaciones sin error", "Valor": f"{convs_exitosas}/{len(conversaciones)} ({(convs_exitosas / len(conversaciones)):.1%})"},
        {"Métrica": "Total de turnos evaluados", "Valor": str(total_turnos)},
        {"Métrica": "Turnos con intención correcta", "Valor": f"{aciertos_totales}/{total_turnos}"},
        {"Métrica": "Precisión global (Accuracy)", "Valor": f"{precision_global:.1%}"},
        {"Métrica": "Tasa de Fallback controlado", "Valor": f"{tasa_fallback:.1%}"},
        {"Métrica": "Eventos registrados en bitácora", "Valor": str(len(bitacora_total))}
    ])

    # 3. Evaluación de Pruebas de Límites
    registro_limites = []
    for item_lim in pruebas_limite:
        caso = item_lim.get("caso", "")
        mensaje = item_lim.get("usuario", "")
        deseada = item_lim.get("intencion_esperada", "")

        session_lim = AuditoriaSession(responsable="Inspector_Limite")
        dialogue_lim = DialogueManager(classifier=chatbot, session=session_lim, ruta_bitacora=None)
        dialogue_lim.responder(mensaje)
        detectada = dialogue_lim.ultima_intencion
        pasa = (detectada == deseada)

        registro_limites.append({
            "Caso de Límite": caso,
            "Mensaje Inspector": mensaje,
            "Deseada": deseada,
            "Detectada": detectada,
            "Resultado": "Superada" if pasa else "Limitación Reglas Léxicas"
        })

    df_limites = pd.DataFrame(registro_limites)

    # Guardar bitácora consolidada en JSON
    Path(ruta_bitacora).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_bitacora, "w", encoding="utf-8") as f:
        json.dump(bitacora_total, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"\nBitácora transaccional exportada en: {ruta_bitacora} ({len(bitacora_total)} registros)")
        print("=" * 80 + "\n")

    return {
        "df_metricas": df_metricas,
        "df_detalle": df_detalle,
        "df_limites": df_limites,
        "precision": precision_global,
        "tasa_fallback": tasa_fallback,
        "total_turnos": total_turnos
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Ejecutor de pruebas del Asistente de Auditoría")
    parser.add_argument("--modo", choices=["reglas", "ml", "ambos"], default="ambos", help="Motor a evaluar ('reglas', 'ml' o 'ambos').")
    args = parser.parse_args()

    if args.modo in ("reglas", "ambos"):
        print("\n" + "#" * 80)
        print("  EVALUACIÓN 1: MOTOR DE REGLAS SIMBÓLICAS (AVANCE 1)")
        print("#" * 80)
        ejecutar_evaluacion(clasificador=RuleBasedChatbot("data/intents.json"), nombre_motor="Reglas Simbólicas (Avance 1)")

    if args.modo in ("ml", "ambos"):
        print("\n" + "#" * 80)
        print("  EVALUACIÓN 2: MOTOR DE MACHINE LEARNING (AVANCE 2 - BoW + LinearSVC)")
        print("#" * 80)
        ejecutar_evaluacion(clasificador=MLIntentClassifier(), nombre_motor="Machine Learning BoW + LinearSVC (Avance 2)")
