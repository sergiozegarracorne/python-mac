"""
Prueba de carga para el proyecto SEDAPAL: genera N frases (variantes no
vistas en el entrenamiento) para los 5 intents (consulta_estado_nis,
consulta_nis_por_medidor, informe_nis, saludo, despedida) y las corre
contra el modelo ya entrenado, mostrando el log en la terminal.

Uso: python test_sedapal.py [cantidad_total]
"""
import random
import string
import sys
from datetime import datetime

from nlu_engine import NluEngine

random.seed(123)


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


def random_nis():
    return str(random.randint(1_000_000, 9_999_999))


def random_medidor():
    letters = "".join(random.choice(string.ascii_uppercase) for _ in range(2))
    digits = "".join(random.choice(string.digits) for _ in range(9))
    return letters + digits


TEMPLATES = {
    "consulta_estado_nis": [
        "cual es el saldo actual del nis {nis}",
        "quiero revisar la deuda del nis {nis}",
        "el nis {nis} esta al dia con los pagos?",
        "necesito el estado de cuenta del suministro {nis}",
        "me puede indicar la situacion del nis {nis}",
        "cuanto se debe en el nis {nis}",
        "el suministro {nis} tiene algo pendiente?",
        "consulteme por favor el nis {nis}, quiero saber si esta al dia",
        "buenas, el nis {nis} cual es su saldo",
        "quisiera verificar el estado del suministro {nis}",
        "el cliente pregunta si el nis {nis} esta al dia",
        "podria revisar cuanto debe el nis {nis}",
        "estado de cuenta suministro {nis} porfavor",
        "el nis {nis}, como va su situacion de pago",
        "deme el saldo pendiente del nis {nis}",
    ],
    "consulta_nis_por_medidor": [
        "tengo el medidor {medidor}, dame el nis por favor",
        "el numero de medidor es {medidor}, cual es el suministro",
        "con el medidor {medidor} me puede dar el nis",
        "busco el nis correspondiente al medidor {medidor}",
        "medidor {medidor}, necesito saber el nis",
        "el cliente solo tiene el medidor {medidor}, busca el nis",
        "identifica el suministro con el medidor {medidor}",
        "segun el medidor {medidor}, cual es el nis del cliente",
        "el medidor instalado tiene el codigo {medidor}, dame el nis",
        "revisando el medidor {medidor} cual es el nis",
        "tengo apuntado el medidor {medidor}, cual es el suministro asociado",
        "solo cuento con el medidor {medidor}, necesito ubicar el nis",
    ],
    "informe_nis": [
        "el nis {nis} ya fue atendido por la contratista",
        "se genero la ot para el nis {nis}",
        "informo que el nis {nis} quedo resuelto",
        "reitero el reclamo del nis {nis} a la contratista",
        "el nis {nis} sigue pendiente de visita",
        "anularon la ot del nis {nis}",
        "confirmado, el nis {nis} fue reparado hoy",
        "el nis {nis} esta de baja en el sistema",
        "ya se coordino la visita para el nis {nis}",
        "el supervisor cerro el caso del nis {nis}",
        "se reasigno el nis {nis} a otra cuadrilla",
        "nis {nis} atendido conforme, sin observaciones",
        "la contratista ya reviso el nis {nis} en terreno",
        "quedo registrada la atencion del nis {nis}",
    ],
}

EXTRA = {
    "saludo": [
        "hola buenas",
        "buenas, como estan",
        "buen dia",
        "hola que tal",
        "aló, buenas tardes",
        "buenas noches, como andan",
    ],
    "despedida": [
        "listo muchas gracias",
        "gracias, eso era todo",
        "perfecto, hasta luego",
        "ok gracias, chau",
        "nos vemos, gracias por la ayuda",
        "listo, nada mas por ahora",
    ],
}


def build_cases(total=300):
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
            if intent == "consulta_nis_por_medidor":
                value = random_medidor()
                text = template.format(medidor=value)
            else:
                value = random_nis()
                text = template.format(nis=value)
            cases.append((text, intent, value))

    for intent in extra_intents:
        phrases = EXTRA[intent]
        for i in range(n_extra):
            text = phrases[i % len(phrases)]
            cases.append((text, intent, None))

    while len(cases) < total:
        intent = random.choice(main_intents)
        template = random.choice(TEMPLATES[intent])
        if intent == "consulta_nis_por_medidor":
            value = random_medidor()
            text = template.format(medidor=value)
        else:
            value = random_nis()
            text = template.format(nis=value)
        cases.append((text, intent, value))

    cases = cases[:total]
    random.shuffle(cases)
    return cases


def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    engine = NluEngine("sedapal", log_callback=log)
    log("Cargando modelo entrenado (proyecto sedapal)...")
    engine.ensure_loaded()

    cases = build_cases(total)
    log(f"Se generaron {len(cases)} frases de prueba. Arrancando analisis...")
    log("-" * 70)

    correct = 0
    entity_ok = 0
    entity_total = 0
    per_intent = {}
    fails = []

    for idx, (text, expected_intent, expected_value) in enumerate(cases, start=1):
        result = engine.parse(text)
        intent = result.get("intent", {}) or {}
        intent_name = intent.get("name") or "desconocida"
        confidence = intent.get("confidence") or 0.0
        entities = result.get("entities", []) or []

        entity_type = "medidor" if expected_intent == "consulta_nis_por_medidor" else "nis"
        values = list(dict.fromkeys(
            e["value"] for e in entities if e.get("entity") == entity_type
        ))

        intent_hit = intent_name == expected_intent
        if intent_hit:
            correct += 1
        per_intent.setdefault(expected_intent, {"total": 0, "hits": 0})
        per_intent[expected_intent]["total"] += 1
        per_intent[expected_intent]["hits"] += int(intent_hit)

        entity_hit = True
        if expected_value is not None:
            entity_total += 1
            entity_hit = expected_value in values
            entity_ok += int(entity_hit)

        mark = "OK " if intent_hit and entity_hit else "FAIL"
        if mark == "FAIL":
            fails.append(
                f"texto='{text}' -> intent={intent_name} ({confidence * 100:.1f}%) "
                f"esperado={expected_intent} valor_detectado={values or '-'} "
                f"esperado_valor={expected_value}"
            )

        if idx % 30 == 0 or mark == "FAIL":
            log(
                f"[{idx:04d}/{len(cases)}] {mark} texto='{text}' "
                f"-> intent={intent_name} ({confidence * 100:.1f}%) "
                f"esperado={expected_intent} valor={values or '-'}"
            )

    log("-" * 70)
    log(f"RESULTADO: {correct}/{len(cases)} intents correctos "
        f"({correct / len(cases) * 100:.2f}%)")
    log(f"RESULTADO ENTIDAD (nis/medidor): {entity_ok}/{entity_total} correctos "
        f"({(entity_ok / entity_total * 100) if entity_total else 0:.2f}%)")
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
