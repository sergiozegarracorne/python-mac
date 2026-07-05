"""
TEST SEDAPAL PRO
================

Prueba robusta del modelo NLU con:
- Lenguaje formal e informal peruano
- Jergas
- Errores ortográficos
- Mensajes cortos
- Mayúsculas/minúsculas
- Puntuación irregular
- NIS: exactamente 7 dígitos
- Medidor: 1 letra + 10 números = 11 caracteres
- Frases ambiguas
- Frases fuera de dominio
- Métricas por intent
- Métricas por tipo de ruido
- Matriz de confusión
- Ranking de errores
- Latencia del modelo

Uso:
    python test_sedapal_pro.py
    python test_sedapal_pro.py 1000
"""

import random
import re
import string
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

from nlu_engine import NluEngine


# ============================================================
# CONFIGURACIÓN
# ============================================================

SEED = 123
DEFAULT_TOTAL = 1000

random.seed(SEED)


# ============================================================
# LOG
# ============================================================

def log(msg=""):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


# ============================================================
# GENERADORES DE ENTIDADES
# ============================================================

def random_nis():
    """
    NIS válido:
    exactamente 7 dígitos.
    """
    return str(random.randint(1_000_000, 9_999_999))


def random_medidor():
    """
    Medidor válido:
    exactamente 11 caracteres.
    1 letra + 10 números.

    Ejemplo:
        A1234567890
    """
    letter = random.choice(string.ascii_uppercase)
    digits = "".join(random.choice(string.digits) for _ in range(10))
    return letter + digits


def is_valid_nis(value):
    return bool(re.fullmatch(r"\d{7}", str(value)))


def is_valid_medidor(value):
    return bool(re.fullmatch(r"[A-Za-z]\d{10}", str(value)))


# ============================================================
# PLANTILLAS BASE
# ============================================================

TEMPLATES = {

    # --------------------------------------------------------
    # CONSULTA DEUDA / ESTADO DEL NIS
    # --------------------------------------------------------

    "consulta_estado_nis": [

        # Formal
        "cual es el saldo actual del nis {nis}",
        "quiero revisar la deuda del nis {nis}",
        "el nis {nis} esta al dia",
        "necesito el estado de cuenta del suministro {nis}",
        "indiqueme la situacion del nis {nis}",
        "cuanto se debe en el nis {nis}",
        "el suministro {nis} tiene deuda",
        "podria revisar cuanto debe el nis {nis}",
        "deme el saldo pendiente del nis {nis}",
        "quiero consultar el estado del suministro {nis}",

        # Normal
        "cuanto debe el nis {nis}",
        "revisa el nis {nis}",
        "consulta el nis {nis}",
        "averigua el nis {nis}",
        "el {nis} debe algo",
        "hay deuda en {nis}",
        "tiene deuda {nis}",
        "saldo del {nis}",
        "deuda del {nis}",
        "estado del nis {nis}",

        # Conversacional
        "quiero saber cuanto debe {nis}",
        "quiero saber si {nis} esta al dia",
        "puedes ver si debe algo {nis}",
        "me dices cuanto debe {nis}",
        "revisame este nis {nis}",
        "consultame este suministro {nis}",
        "fijate si tiene deuda {nis}",
        "dime si debe algo {nis}",

        # Peruano informal
        "oe revisa el nis {nis}",
        "oe cuanto debe {nis}",
        "causa revisame el {nis}",
        "mano consulta este nis {nis}",
        "broder cuanto debe el {nis}",
        "porfa revisa el {nis}",
        "revisalo pe {nis}",
        "aver cuanto debe este {nis}",
        "chequea este nis {nis} porfa",
        "oe mira si debe algo {nis}",
        "ya pe revisa el {nis}",
        "consulta esto pe {nis}",

        # Muy corto
        "deuda {nis}",
        "saldo {nis}",
        "nis {nis}",
        "{nis} deuda",
        "{nis} saldo",
        "{nis} debe",
        "{nis} esta al dia",
        "consulta {nis}",
        "revisa {nis}",

        # Errores comunes
        "kiero saber la deuda de {nis}",
        "quiero saver cuanto deve {nis}",
        "aver si tiene deuda {nis}",
        "cuanto deve el nis {nis}",
        "revisame el suministo {nis}",
        "consulta la deudaa de {nis}",
        "cuanto deb el {nis}",
        "el nis {nis} tien deuda",
        "kisiera saber el saldo {nis}",
        "nesecito saber cuanto debe {nis}",
        "porfabor revisa {nis}",
    ],

    # --------------------------------------------------------
    # BUSCAR NIS POR MEDIDOR
    # --------------------------------------------------------

    "consulta_nis_por_medidor": [

        # Formal
        "tengo el medidor {medidor}, indiqueme el nis",
        "el numero de medidor es {medidor}, cual es el suministro",
        "con el medidor {medidor} puede obtener el nis",
        "busco el nis correspondiente al medidor {medidor}",
        "identifique el suministro del medidor {medidor}",
        "cual es el nis asociado al medidor {medidor}",

        # Normal
        "tengo el medidor {medidor} dame el nis",
        "medidor {medidor} cual es su nis",
        "busca el nis de {medidor}",
        "dame el nis del medidor {medidor}",
        "consulta este medidor {medidor}",
        "que nis tiene {medidor}",
        "cual es el suministro de {medidor}",
        "ubica el nis con {medidor}",

        # Conversacional
        "solo tengo el medidor {medidor}",
        "no tengo nis solo el medidor {medidor}",
        "con esto puedes sacar el nis {medidor}",
        "quiero saber el nis de este medidor {medidor}",
        "me ayudas a encontrar el nis con {medidor}",
        "tengo este codigo de medidor {medidor}",

        # Peruano informal
        "oe tengo el medidor {medidor} saca el nis",
        "causa busca el nis de {medidor}",
        "mano solo tengo este medidor {medidor}",
        "broder dame el nis de {medidor}",
        "aver sacame el nis con {medidor}",
        "porfa busca el nis de este medidor {medidor}",
        "oe ubica el suministro con {medidor}",
        "ya pe busca el nis {medidor}",

        # Muy corto
        "medidor {medidor}",
        "{medidor} nis",
        "nis de {medidor}",
        "buscar {medidor}",
        "consulta medidor {medidor}",
        "{medidor} suministro",

        # Errores comunes
        "medior {medidor} cual es el nis",
        "medidro {medidor} dame el nis",
        "kiero el nis de {medidor}",
        "buska el nis de {medidor}",
        "cual es el niz de {medidor}",
        "sacame el nis del medior {medidor}",
        "nesecito el nis de {medidor}",
        "averigua el suministo de {medidor}",
    ],

    # --------------------------------------------------------
    # INFORMES / REPORTES OPERATIVOS
    # --------------------------------------------------------

    "informe_nis": [

        # Atendido
        "el nis {nis} ya fue atendido por la contratista",
        "el nis {nis} fue atendido",
        "nis {nis} atendido conforme",
        "ya atendieron el nis {nis}",
        "la contratista atendio el {nis}",
        "trabajo terminado en {nis}",

        # Resuelto
        "informo que el nis {nis} quedo resuelto",
        "el problema del nis {nis} fue solucionado",
        "caso resuelto para el nis {nis}",
        "ya solucionaron el {nis}",
        "el nis {nis} quedo operativo",

        # OT
        "se genero la ot para el nis {nis}",
        "se creo una ot para {nis}",
        "la ot del nis {nis} fue anulada",
        "anularon la ot de {nis}",
        "ot generada para {nis}",

        # Pendiente
        "el nis {nis} sigue pendiente",
        "el nis {nis} sigue pendiente de visita",
        "aun no atienden el nis {nis}",
        "falta atender el {nis}",
        "todavia esta pendiente {nis}",

        # Coordinación
        "se coordino la visita para el nis {nis}",
        "visita coordinada para {nis}",
        "se programo visita al nis {nis}",
        "la cuadrilla ira al nis {nis}",

        # Reasignación
        "se reasigno el nis {nis} a otra cuadrilla",
        "el caso {nis} paso a otra cuadrilla",
        "reasignaron la atencion del nis {nis}",

        # Terreno
        "la contratista reviso el nis {nis} en terreno",
        "se realizo inspeccion del nis {nis}",
        "el personal estuvo en el nis {nis}",

        # Peruano informal
        "oe ya atendieron el {nis}",
        "causa el {nis} ya quedo",
        "mano ya solucionaron el {nis}",
        "el {nis} sigue pendiente pe",
        "todavia nada con el {nis}",
        "ya fueron al {nis}",
        "ya lo arreglaron {nis}",
        "el {nis} ya esta listo",
        "ya cerraron el caso {nis}",

        # Errores comunes
        "el nis {nis} ya fue atendio",
        "el {nis} ya esta resueltoo",
        "todabia falta atender {nis}",
        "se genero la ot para el niz {nis}",
        "ya lo areglaron {nis}",
        "la contratista ya atendioo {nis}",
        "el caso {nis} sige pendiente",
        "ya quedo solusionado {nis}",
    ],
}


# ============================================================
# SALUDOS Y DESPEDIDAS
# ============================================================

EXTRA = {

    "saludo": [
        "hola",
        "holaa",
        "holaaa",
        "ola",
        "ola buenas",
        "holi",
        "buenas",
        "buenass",
        "buen dia",
        "buenos dias",
        "buenas tardes",
        "buenas noches",
        "hola buenas",
        "hola que tal",
        "que tal",
        "como estan",
        "alo",
        "aló",
        "oe",
        "oe buenas",
        "habla",
        "habla causa",
        "habla mano",
        "que fue",
        "que tal causa",
        "buenas broder",
        "hola amigo",
        "hola porfa una consulta",
        "buenas tengo una consulta",
    ],

    "despedida": [
        "gracias",
        "grasias",
        "graciaas",
        "listo gracias",
        "muchas gracias",
        "gracias eso era todo",
        "perfecto hasta luego",
        "ok gracias",
        "ok chau",
        "chau",
        "chao",
        "nos vemos",
        "listo nada mas",
        "eso es todo",
        "ya gracias",
        "ya fue gracias",
        "bacan gracias",
        "chevere gracias",
        "gracias causa",
        "gracias mano",
        "gracias broder",
        "listo pe gracias",
        "ya quedo",
        "perfecto",
        "todo bien gracias",
    ],
}


# ============================================================
# MODIFICADORES DE RUIDO
# ============================================================

FILLERS_START = [
    "",
    "",
    "",
    "por favor ",
    "porfa ",
    "oe ",
    "oye ",
    "causa ",
    "mano ",
    "broder ",
    "amigo ",
    "una consulta ",
    "consulta ",
    "aver ",
    "a ver ",
]

FILLERS_END = [
    "",
    "",
    "",
    " por favor",
    " porfa",
    " pe",
    " pues",
    " gracias",
    " mano",
    " causa",
    " nomas",
    " rapidito",
]

TYPO_REPLACEMENTS = [
    ("quiero", "kiero"),
    ("quiero", "qiero"),
    ("saber", "saver"),
    ("debe", "deve"),
    ("deuda", "deudaa"),
    ("medidor", "medior"),
    ("medidor", "medidro"),
    ("suministro", "suministo"),
    ("necesito", "nesecito"),
    ("buscar", "buskar"),
    ("buscar", "buska"),
    ("porque", "xq"),
    ("por favor", "porfa"),
    ("tambien", "tmb"),
    ("esta", "ta"),
    ("para", "pa"),
]


def remove_accents_simple(text):
    replacements = {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "Á": "A",
        "É": "E",
        "Í": "I",
        "Ó": "O",
        "Ú": "U",
    }

    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    return text


def add_typo(text):
    candidates = [
        (a, b)
        for a, b in TYPO_REPLACEMENTS
        if a in text.lower()
    ]

    if not candidates:
        return text

    original, replacement = random.choice(candidates)

    return re.sub(
        re.escape(original),
        replacement,
        text,
        count=1,
        flags=re.IGNORECASE,
    )


def remove_random_vowel(text):
    """
    Simula errores:
        quiero -> quier
        consulta -> conslta

    Nunca modifica números ni códigos.
    """
    words = text.split()

    candidates = [
        i
        for i, word in enumerate(words)
        if len(word) >= 6
        and not re.search(r"\d", word)
    ]

    if not candidates:
        return text

    index = random.choice(candidates)
    word = words[index]

    vowel_positions = [
        i
        for i, char in enumerate(word)
        if char.lower() in "aeiou"
        and i > 0
    ]

    if vowel_positions:
        pos = random.choice(vowel_positions)
        words[index] = word[:pos] + word[pos + 1:]

    return " ".join(words)


def mutate_case(text):
    option = random.choice([
        "lower",
        "lower",
        "lower",
        "upper",
        "capitalize",
    ])

    if option == "upper":
        return text.upper()

    if option == "capitalize":
        return text.capitalize()

    return text.lower()


def mutate_punctuation(text):
    option = random.choice([
        "none",
        "none",
        "question",
        "double_question",
        "dots",
        "comma",
    ])

    if option == "question":
        return text + "?"

    if option == "double_question":
        return text + "??"

    if option == "dots":
        return text + "..."

    if option == "comma":
        return text.replace(" ", ", ", 1)

    return text


def apply_noise(text):
    """
    Aplica ruido controlado y devuelve:
        texto_modificado, tipo_de_ruido
    """

    noise = random.choice([
        "limpio",
        "limpio",
        "jerga",
        "typo",
        "sin_vocal",
        "mayusculas",
        "puntuacion",
        "combinado",
    ])

    if noise == "limpio":
        return text, noise

    if noise == "jerga":
        text = random.choice(FILLERS_START) + text
        text += random.choice(FILLERS_END)

    elif noise == "typo":
        text = add_typo(text)

    elif noise == "sin_vocal":
        text = remove_random_vowel(text)

    elif noise == "mayusculas":
        text = mutate_case(text)

    elif noise == "puntuacion":
        text = mutate_punctuation(text)

    elif noise == "combinado":
        text = random.choice(FILLERS_START) + text
        text += random.choice(FILLERS_END)

        if random.random() < 0.7:
            text = add_typo(text)

        if random.random() < 0.3:
            text = remove_random_vowel(text)

        text = mutate_case(text)
        text = mutate_punctuation(text)

    return text.strip(), noise


# ============================================================
# GENERACIÓN DE CASOS
# ============================================================

def build_case(intent):
    if intent in TEMPLATES:

        template = random.choice(TEMPLATES[intent])

        if intent == "consulta_nis_por_medidor":
            value = random_medidor()

            assert is_valid_medidor(value), (
                f"Medidor inválido generado: {value}"
            )

            text = template.format(medidor=value)

        else:
            value = random_nis()

            assert is_valid_nis(value), (
                f"NIS inválido generado: {value}"
            )

            text = template.format(nis=value)

        text, noise = apply_noise(text)

        return {
            "text": text,
            "expected_intent": intent,
            "expected_value": value,
            "noise": noise,
        }

    text = random.choice(EXTRA[intent])
    text, noise = apply_noise(text)

    return {
        "text": text,
        "expected_intent": intent,
        "expected_value": None,
        "noise": noise,
    }


def build_cases(total):
    intents = list(TEMPLATES.keys()) + list(EXTRA.keys())

    cases = []

    # Distribución equilibrada
    while len(cases) < total:
        for intent in intents:

            if len(cases) >= total:
                break

            cases.append(build_case(intent))

    random.shuffle(cases)

    return cases


# ============================================================
# EXTRACCIÓN DE ENTIDADES
# ============================================================

def extract_entity_values(entities, expected_intent):

    if expected_intent == "consulta_nis_por_medidor":
        entity_type = "medidor"

    elif expected_intent in (
        "consulta_estado_nis",
        "informe_nis",
    ):
        entity_type = "nis"

    else:
        return []

    return list(dict.fromkeys(
        str(entity.get("value"))
        for entity in entities
        if entity.get("entity") == entity_type
    ))


# ============================================================
# MATRIZ DE CONFUSIÓN
# ============================================================

def print_confusion_matrix(confusion, intents):

    log("")
    log("MATRIZ DE CONFUSIÓN")
    log("-" * 100)

    width = 25

    header = "ESPERADO".ljust(width)

    for intent in intents:
        header += intent[:15].center(17)

    log(header)

    for expected in intents:

        row = expected[:24].ljust(width)

        for predicted in intents:
            value = confusion[expected][predicted]
            row += str(value).center(17)

        log(row)


# ============================================================
# MAIN
# ============================================================

def main():

    total = (
        int(sys.argv[1])
        if len(sys.argv) > 1
        else DEFAULT_TOTAL
    )

    log("=" * 100)
    log("SEDAPAL NLU - TEST PRO")
    log("=" * 100)

    engine = NluEngine(
        "sedapal",
        log_callback=log,
    )

    log("Cargando modelo entrenado...")

    start_load = time.perf_counter()

    engine.ensure_loaded()

    load_time = time.perf_counter() - start_load

    log(f"Modelo cargado en {load_time:.3f} segundos")

    cases = build_cases(total)

    log(f"Casos generados: {len(cases)}")
    log("Iniciando análisis...")
    log("-" * 100)

    correct_intents = 0

    entity_ok = 0
    entity_total = 0

    per_intent = defaultdict(
        lambda: {
            "total": 0,
            "hits": 0,
        }
    )

    per_noise = defaultdict(
        lambda: {
            "total": 0,
            "hits": 0,
        }
    )

    confusion = defaultdict(Counter)

    fails = []

    latencies = []

    all_intents = (
        list(TEMPLATES.keys())
        + list(EXTRA.keys())
    )

    for idx, case in enumerate(cases, start=1):

        text = case["text"]
        expected_intent = case["expected_intent"]
        expected_value = case["expected_value"]
        noise = case["noise"]

        start = time.perf_counter()

        result = engine.parse(text)

        latency = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(latency)

        intent_data = result.get("intent") or {}

        detected_intent = (
            intent_data.get("name")
            or "desconocida"
        )

        confidence = float(
            intent_data.get("confidence")
            or 0.0
        )

        entities = result.get("entities") or []

        values = extract_entity_values(
            entities,
            expected_intent,
        )

        # ----------------------------------------------------
        # INTENT
        # ----------------------------------------------------

        intent_hit = (
            detected_intent == expected_intent
        )

        if intent_hit:
            correct_intents += 1

        per_intent[expected_intent]["total"] += 1
        per_intent[expected_intent]["hits"] += int(intent_hit)

        per_noise[noise]["total"] += 1
        per_noise[noise]["hits"] += int(intent_hit)

        confusion[expected_intent][detected_intent] += 1

        # ----------------------------------------------------
        # ENTIDAD
        # ----------------------------------------------------

        entity_hit = True

        if expected_value is not None:

            entity_total += 1

            entity_hit = (
                expected_value.upper()
                in [v.upper() for v in values]
            )

            entity_ok += int(entity_hit)

        # ----------------------------------------------------
        # RESULTADO
        # ----------------------------------------------------

        full_hit = intent_hit and entity_hit

        if not full_hit:

            fails.append({
                "text": text,
                "expected": expected_intent,
                "detected": detected_intent,
                "confidence": confidence,
                "expected_value": expected_value,
                "values": values,
                "noise": noise,
            })

        if idx % 50 == 0 or not full_hit:

            mark = "OK" if full_hit else "FAIL"

            log(
                f"[{idx:05d}/{total}] "
                f"{mark:<4} "
                f"[{noise:<10}] "
                f"'{text}' "
                f"=> {detected_intent} "
                f"({confidence * 100:.1f}%)"
            )

    # ========================================================
    # RESULTADOS
    # ========================================================

    intent_accuracy = (
        correct_intents / total * 100
    )

    entity_accuracy = (
        entity_ok / entity_total * 100
        if entity_total
        else 0
    )

    avg_latency = (
        sum(latencies) / len(latencies)
    )

    max_latency = max(latencies)
    min_latency = min(latencies)

    throughput = (
        1000 / avg_latency
        if avg_latency
        else 0
    )

    log("")
    log("=" * 100)
    log("RESULTADO GENERAL")
    log("=" * 100)

    log(
        f"INTENTS: "
        f"{correct_intents}/{total} "
        f"({intent_accuracy:.2f}%)"
    )

    log(
        f"ENTIDADES: "
        f"{entity_ok}/{entity_total} "
        f"({entity_accuracy:.2f}%)"
    )

    log(
        f"LATENCIA PROMEDIO: "
        f"{avg_latency:.3f} ms"
    )

    log(
        f"LATENCIA MÍNIMA: "
        f"{min_latency:.3f} ms"
    )

    log(
        f"LATENCIA MÁXIMA: "
        f"{max_latency:.3f} ms"
    )

    log(
        f"RENDIMIENTO APROX.: "
        f"{throughput:.1f} frases/segundo"
    )

    # ========================================================
    # RESULTADO POR INTENT
    # ========================================================

    log("")
    log("RESULTADO POR INTENT")
    log("-" * 100)

    for intent_name, stats in per_intent.items():

        pct = (
            stats["hits"]
            / stats["total"]
            * 100
        )

        log(
            f"{intent_name:<30} "
            f"{stats['hits']:>5}/"
            f"{stats['total']:<5} "
            f"{pct:>7.2f}%"
        )

    # ========================================================
    # RESULTADO POR RUIDO
    # ========================================================

    log("")
    log("RESULTADO POR TIPO DE ESCRITURA")
    log("-" * 100)

    for noise, stats in sorted(per_noise.items()):

        pct = (
            stats["hits"]
            / stats["total"]
            * 100
        )

        log(
            f"{noise:<20} "
            f"{stats['hits']:>5}/"
            f"{stats['total']:<5} "
            f"{pct:>7.2f}%"
        )

    # ========================================================
    # MATRIZ DE CONFUSIÓN
    # ========================================================

    print_confusion_matrix(
        confusion,
        all_intents,
    )

    # ========================================================
    # FALLOS
    # ========================================================

    if fails:

        log("")
        log("=" * 100)
        log(f"FALLOS DETECTADOS: {len(fails)}")
        log("=" * 100)

        # Los más seguros pero equivocados primero
        fails.sort(
            key=lambda x: x["confidence"],
            reverse=True,
        )

        for fail in fails[:100]:

            log(
                f"FAIL "
                f"[{fail['noise']}] "
                f"'{fail['text']}' "
                f"=> {fail['detected']} "
                f"({fail['confidence'] * 100:.1f}%) "
                f"| esperado={fail['expected']} "
                f"| entidad={fail['values'] or '-'} "
                f"| esperado_valor="
                f"{fail['expected_value'] or '-'}"
            )

        if len(fails) > 100:
            log(
                f"... se muestran solo los primeros "
                f"100 de {len(fails)} fallos."
            )

    log("")
    log("=" * 100)
    log("TEST FINALIZADO")
    log("=" * 100)


if __name__ == "__main__":
    main()
