"""
Cola de casos pendientes de revision. Cuando el modelo responde con poca
confianza, el caso queda guardado aca (una base sqlite por proyecto) para
que despues un humano indique cual era la intencion correcta. Esa
correccion se puede volcar como ejemplo nuevo en data/nlu.yml, cerrando el
circuito de mejora continua (el modelo aprende de sus propios errores).
"""
import json
import os
import sqlite3
import time

from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString

from nlu_engine import PROJECTS_DIR


def _db_path(project):
    return os.path.join(PROJECTS_DIR, project, "pendientes.db")


def _connect(project):
    conn = sqlite3.connect(_db_path(project))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pendientes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            texto TEXT NOT NULL,
            intent_predicho TEXT,
            confianza REAL,
            entidades TEXT,
            fecha TEXT,
            estado TEXT DEFAULT 'pendiente',
            intent_corregido TEXT,
            entidad_tipo TEXT,
            entidad_valor TEXT
        )
        """
    )
    conn.commit()
    return conn


def registrar_caso(project, texto, intent_predicho, confianza, entidades):
    conn = _connect(project)
    conn.execute(
        "INSERT INTO pendientes (texto, intent_predicho, confianza, entidades, fecha) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            texto,
            intent_predicho,
            confianza,
            json.dumps(entidades),
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    conn.commit()
    conn.close()


def listar_pendientes(project):
    conn = _connect(project)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM pendientes WHERE estado = 'pendiente' ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def contar_pendientes(project):
    conn = _connect(project)
    (count,) = conn.execute(
        "SELECT COUNT(*) FROM pendientes WHERE estado = 'pendiente'"
    ).fetchone()
    conn.close()
    return count


def marcar_ignorado(project, caso_id):
    conn = _connect(project)
    conn.execute("UPDATE pendientes SET estado='ignorado' WHERE id=?", (caso_id,))
    conn.commit()
    conn.close()


def marcar_revisado(project, caso_id, intent_corregido, entidad_tipo, entidad_valor):
    conn = _connect(project)
    conn.execute(
        "UPDATE pendientes SET estado='revisado', intent_corregido=?, "
        "entidad_tipo=?, entidad_valor=? WHERE id=?",
        (intent_corregido, entidad_tipo, entidad_valor, caso_id),
    )
    conn.commit()
    conn.close()


def agregar_ejemplo_a_nlu(project, intent_name, texto, entidad_tipo=None, entidad_valor=None):
    """Suma `texto` (con la entidad anotada [valor](tipo) si corresponde) como
    ejemplo nuevo del intent indicado, directo en data/nlu.yml."""
    path = os.path.join(PROJECTS_DIR, project, "data", "nlu.yml")

    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.load(f)

    frase = texto
    if entidad_tipo and entidad_valor and entidad_valor in texto:
        frase = texto.replace(entidad_valor, f"[{entidad_valor}]({entidad_tipo})", 1)

    for item in data["nlu"]:
        if item.get("intent") == intent_name:
            actuales = str(item["examples"])
            nuevas = actuales.rstrip("\n") + f"\n- {frase}\n"
            item["examples"] = LiteralScalarString(nuevas)
            break
    else:
        raise ValueError(f"No se encontro el intent '{intent_name}' en nlu.yml")

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f)

    return frase
