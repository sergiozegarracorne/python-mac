"""
Prueba de carga: genera 100 frases (variantes no vistas en el entrenamiento)
para los 3 intents principales + saludo/despedida, las corre contra el
modelo Rasa NLU ya entrenado, e imprime el log en la terminal igual que
lo haria el panel de logs del formulario.
"""
import random
import sys
from datetime import datetime

from nlu_engine import NluEngine

random.seed(7)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


TEMPLATES = {
    "consultar_reporte_cliente": [
        "necesito el reporte del cliente {dni}",
        "podrias darme el reporte del cliente {dni}",
        "quiero consultar el reporte del cliente {dni}",
        "buscame el reporte del cliente numero {dni}",
        "el cliente {dni} pidio su reporte",
        "envia el reporte del cliente {dni}",
        "activa el reporte para el cliente {dni}",
        "quiero revisar el reporte de cliente {dni}",
        "hazme el reporte del cliente {dni}",
        "cliente {dni} solicita reporte",
    ],
    "consultar_estado_cuenta": [
        "quiero el estado de cuenta cliente {dni}",
        "revisame el estado de cuenta del cliente {dni}",
        "consulta estado de cuenta del cliente {dni}",
        "el cliente {dni} pregunta por su estado de cuenta",
        "pasame estado de cuenta cliente:{dni}",
        "actualizame el estado de cuenta del cliente {dni}",
        "quiero saber el estado de cuenta de {dni}",
        "chequea el estado de cuenta del cliente {dni}",
        "el estado de cuenta de {dni} por favor",
        "necesito revisar estado de cuenta cliente {dni}",
    ],
    "consultar_reporte_deudas": [
        "quiero el reporte de deudas de {dni}",
        "el cliente {dni} pregunta por sus deudas",
        "cuanto debe el cliente {dni}",
        "dame las deudas pendientes del cliente {dni}",
        "informe de deudas cliente {dni}",
        "el cliente {dni} necesita su reporte de deudas",
        "consulta de morosidad del cliente {dni}",
        "revisa las deudas del dni {dni}",
        "cliente {dni} quiere saber sus deudas",
        "traeme deudas pendientes dni {dni}",
    ],
}

EXTRA = {
    "saludo": [
        "hola como estas",
        "buenas, necesito ayuda",
        "hola buen dia",
        "que tal todo bien",
        "buenas tardes",
        "hola que haces",
        "buenas noches",
        "hola, como andas",
    ],
    "despedida": [
        "listo gracias",
        "eso es todo, gracias",
        "chau nos vemos",
        "perfecto, hasta luego",
        "ok gracias adios",
        "muchas gracias, chau",
        "listo, nada mas",
        "gracias, nos vemos",
    ],
}


def build_cases(n_per_intent=28):
    cases = []
    for intent, templates in TEMPLATES.items():
        for i in range(n_per_intent):
            template = templates[i % len(templates)]
            dni = str(random.randint(10_000_000, 99_999_999))
            text = template.format(dni=dni)
            cases.append((text, intent, dni))
    for intent, phrases in EXTRA.items():
        for text in phrases:
            cases.append((text, intent, None))
    random.shuffle(cases)
    return cases


def main():
    engine = NluEngine("reporte_cliente", log_callback=log)
    log("Cargando modelo entrenado...")
    engine.ensure_loaded()

    cases = build_cases()
    log(f"Se generaron {len(cases)} frases de prueba. Arrancando analisis...")
    log("-" * 70)

    correct = 0
    dni_ok = 0
    dni_total = 0
    per_intent = {}

    for idx, (text, expected_intent, expected_dni) in enumerate(cases, start=1):
        result = engine.parse(text)
        intent = result.get("intent", {}) or {}
        intent_name = intent.get("name") or "desconocida"
        confidence = intent.get("confidence") or 0.0
        entities = result.get("entities", []) or []
        dnis = list(dict.fromkeys(e["value"] for e in entities if e.get("entity") == "dni"))

        intent_hit = intent_name == expected_intent
        if intent_hit:
            correct += 1
        per_intent.setdefault(expected_intent, {"total": 0, "hits": 0})
        per_intent[expected_intent]["total"] += 1
        per_intent[expected_intent]["hits"] += int(intent_hit)

        dni_hit = True
        if expected_dni is not None:
            dni_total += 1
            dni_hit = expected_dni in dnis
            dni_ok += int(dni_hit)

        mark = "OK " if intent_hit and dni_hit else "FAIL"
        log(
            f"[{idx:03d}/{len(cases)}] {mark} texto='{text}' "
            f"-> intent={intent_name} ({confidence * 100:.1f}%) "
            f"esperado={expected_intent} dni_detectado={dnis or '-'}"
        )

    log("-" * 70)
    log(f"RESULTADO: {correct}/{len(cases)} intents correctos "
        f"({correct / len(cases) * 100:.1f}%)")
    log(f"RESULTADO DNI: {dni_ok}/{dni_total} DNIs correctamente extraidos "
        f"({(dni_ok / dni_total * 100) if dni_total else 0:.1f}%)")
    for intent_name, stats in per_intent.items():
        pct = stats["hits"] / stats["total"] * 100
        log(f"  - {intent_name}: {stats['hits']}/{stats['total']} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
