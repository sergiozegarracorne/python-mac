"""
Prueba de carga: genera 1000 frases (variantes no vistas en el entrenamiento)
para los 3 intents principales + saludo/despedida, las corre contra el
modelo Rasa NLU ya entrenado, e imprime el log en la terminal igual que
lo haria el panel de logs del formulario.

Uso: python test_1000.py [cantidad_total]
"""
import random
import sys
from datetime import datetime

from nlu_engine import NluEngine

random.seed(42)


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
        "abrime el reporte del cliente {dni}",
        "quiero el informe del cliente {dni}",
        "generame el reporte completo del cliente {dni}",
        "revisar reporte cliente {dni} por favor",
        "el cliente con documento {dni} pide su reporte",
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
        "traeme el estado de cuenta cliente {dni}",
        "el cliente {dni} quiere ver su estado de cuenta",
        "abrime el estado de cuenta del cliente {dni}",
        "consulta rapida de estado de cuenta cliente {dni}",
        "quiero el resumen de cuenta del cliente {dni}",
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
        "cuanto adeuda el cliente {dni}",
        "decime cuanto debe el cliente {dni}",
        "a cuanto asciende la deuda del cliente {dni}",
        "quiero saber la deuda total del cliente {dni}",
        "el cliente {dni} pregunta cuanto debe",
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
        "que tal, todo en orden?",
        "hola, buenas",
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
        "eso era todo, adios",
        "listo, muchas gracias, chau",
    ],
}


def build_cases(total=1000):
    main_intents = list(TEMPLATES.keys())
    extra_intents = list(EXTRA.keys())

    n_extra = max(1, total // 20)
    n_main_total = total - n_extra * len(extra_intents)
    n_per_intent = n_main_total // len(main_intents)

    cases = []
    for intent in main_intents:
        templates = TEMPLATES[intent]
        for i in range(n_per_intent):
            template = templates[i % len(templates)]
            dni = str(random.randint(10_000_000, 99_999_999))
            text = template.format(dni=dni)
            cases.append((text, intent, dni))

    for intent in extra_intents:
        phrases = EXTRA[intent]
        for i in range(n_extra):
            text = phrases[i % len(phrases)]
            cases.append((text, intent, None))

    # completar redondeos hasta llegar exacto a `total`
    while len(cases) < total:
        intent = random.choice(main_intents)
        template = random.choice(TEMPLATES[intent])
        dni = str(random.randint(10_000_000, 99_999_999))
        cases.append((template.format(dni=dni), intent, dni))

    cases = cases[:total]
    random.shuffle(cases)
    return cases


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    engine = NluEngine("reporte_cliente", log_callback=log)
    log("Cargando modelo entrenado...")
    engine.ensure_loaded()

    cases = build_cases(total)
    log(f"Se generaron {len(cases)} frases de prueba. Arrancando analisis...")
    log("-" * 70)

    correct = 0
    dni_ok = 0
    dni_total = 0
    per_intent = {}
    fails = []

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
        if mark == "FAIL":
            fails.append(
                f"texto='{text}' -> intent={intent_name} ({confidence * 100:.1f}%) "
                f"esperado={expected_intent} dni_detectado={dnis or '-'}"
            )

        if idx % 50 == 0 or mark == "FAIL":
            log(
                f"[{idx:04d}/{len(cases)}] {mark} texto='{text}' "
                f"-> intent={intent_name} ({confidence * 100:.1f}%) "
                f"esperado={expected_intent} dni_detectado={dnis or '-'}"
            )

    log("-" * 70)
    log(f"RESULTADO: {correct}/{len(cases)} intents correctos "
        f"({correct / len(cases) * 100:.2f}%)")
    log(f"RESULTADO DNI: {dni_ok}/{dni_total} DNIs correctamente extraidos "
        f"({(dni_ok / dni_total * 100) if dni_total else 0:.2f}%)")
    for intent_name, stats in per_intent.items():
        pct = stats["hits"] / stats["total"] * 100
        log(f"  - {intent_name}: {stats['hits']}/{stats['total']} ({pct:.1f}%)")

    if fails:
        log("-" * 70)
        log(f"Fallos ({len(fails)}):")
        for f in fails:
            log(f"  FAIL {f}")


if __name__ == "__main__":
    main()
