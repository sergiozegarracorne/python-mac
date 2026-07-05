"""
Formulario moderno para probar/entrenar modelos Rasa NLU. Soporta varios
proyectos (carpetas independientes bajo proyectos/) para poder tener mas
de un bot entrenado sin que se pisen entre si, ej: "reporte_cliente" y
"sedapal".
"""
import json
import queue
import threading
import traceback
from datetime import datetime

import customtkinter as ctk

from nlu_engine import NluEngine, list_projects
import revision_pendientes as revision

CONFIDENCE_THRESHOLD = 0.75

# ---------------------------------------------------------------------------
# Configuracion por proyecto: como mostrar cada intent y que "funcion" de
# negocio (endpoint) se dispararia con el valor detectado.
# ---------------------------------------------------------------------------
PROJECTS = {
    "reporte_cliente": {
        "display_name": "Reporte Cliente",
        "entity_types": ["dni"],
        "intent_labels": {
            "consultar_reporte_cliente": "Reporte del cliente",
            "consultar_estado_cuenta": "Estado de cuenta",
            "consultar_reporte_deudas": "Reporte de deudas",
            "saludo": "Saludo",
            "despedida": "Despedida",
        },
        "function_map": {
            "consultar_reporte_cliente": "CONSULTA_REPORTE_CLIENTE/{value}",
            "consultar_estado_cuenta": "CONSULTA_ESTADO_CUENTA/{value}",
            "consultar_reporte_deudas": "CONSULTA_REPORTE_DEUDAS/{value}",
        },
    },
    "sedapal": {
        "display_name": "SEDAPAL",
        "entity_types": ["nis", "medidor"],
        "intent_labels": {
            "consulta_estado_nis": "Consulta estado del NIS",
            "consulta_nis_por_medidor": "Consulta NIS por medidor",
            "informe_nis": "Informe (NO es una consulta)",
            "saludo": "Saludo",
            "despedida": "Despedida",
        },
        "function_map": {
            "consulta_estado_nis": "CONSULTA_ESTADO_NIS/{value}",
            "consulta_nis_por_medidor": "CONSULTA_NIS_X_MEDIDOR/{value}",
            "informe_nis": "NIS_ALERTA/{value}  (solo notifica, no consulta)",
        },
    },
}

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Entrenador de Intenciones - Rasa NLU")
        self.geometry("1020x760")
        self.minsize(900, 640)

        self.log_queue = queue.Queue()

        available = list_projects() or list(PROJECTS.keys())
        self.current_project = available[0] if available else "reporte_cliente"
        self.engine = NluEngine(self.current_project, log_callback=self.enqueue_log)

        self._build_layout(available)
        self.after(150, self._drain_log_queue)
        self.enqueue_log(
            f"Aplicacion iniciada. Proyecto activo: {self._project_label(self.current_project)}."
        )
        self._refresh_pending_count()

    # ---------- helpers de proyecto ----------
    def _project_label(self, project):
        return PROJECTS.get(project, {}).get("display_name", project)

    def _project_cfg(self):
        return PROJECTS.get(self.current_project, {"intent_labels": {}, "function_map": {}, "entity_types": []})

    # ---------- UI ----------
    def _build_layout(self, available_projects):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        header = ctk.CTkFrame(self, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        title_row = ctk.CTkFrame(header, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew", padx=24, pady=(18, 2))
        title_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            title_row,
            text="Detector de intenciones",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(title_row, text="Proyecto:", font=ctk.CTkFont(size=13)).grid(
            row=0, column=1, sticky="e", padx=(0, 8)
        )
        self.project_menu = ctk.CTkOptionMenu(
            title_row,
            values=[self._project_label(p) for p in available_projects],
            command=self.on_project_change,
            width=180,
        )
        self.project_menu.set(self._project_label(self.current_project))
        self.project_menu.grid(row=0, column=2, sticky="e")

        ctk.CTkLabel(
            header,
            text="Escribi una frase como la escribiria un usuario real y fijate que intencion/parametro detecta el modelo.",
            font=ctk.CTkFont(size=13),
            text_color=("gray20", "gray70"),
        ).grid(row=1, column=0, sticky="w", padx=24, pady=(0, 16))

        # ---- Panel de acciones + resultado ----
        top = ctk.CTkFrame(self, corner_radius=12)
        top.grid(row=1, column=0, sticky="ew", padx=24, pady=(16, 8))
        top.grid_columnconfigure(0, weight=1)

        actions = ctk.CTkFrame(top, fg_color="transparent")
        actions.grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 8))
        actions.grid_columnconfigure(3, weight=1)

        self.train_btn = ctk.CTkButton(
            actions, text="Entrenar modelo", width=160, command=self.on_train
        )
        self.train_btn.grid(row=0, column=0, padx=(0, 8))

        self.load_btn = ctk.CTkButton(
            actions,
            text="Cargar ultimo modelo",
            width=170,
            fg_color="transparent",
            border_width=1,
            command=self.on_load,
        )
        self.load_btn.grid(row=0, column=1, padx=(0, 8))

        self.pending_btn = ctk.CTkButton(
            actions,
            text="Revisar pendientes (0)",
            width=180,
            fg_color="#8a5a00",
            hover_color="#6e4800",
            command=self.open_review_window,
        )
        self.pending_btn.grid(row=0, column=2, padx=(0, 8))

        self.status_label = ctk.CTkLabel(
            actions, text="Estado: sin modelo cargado", text_color=("gray20", "gray70")
        )
        self.status_label.grid(row=0, column=3, sticky="e")

        # Entry + enviar
        entry_row = ctk.CTkFrame(top, fg_color="transparent")
        entry_row.grid(row=1, column=0, sticky="ew", padx=16, pady=(8, 16))
        entry_row.grid_columnconfigure(0, weight=1)

        self.text_entry = ctk.CTkEntry(
            entry_row,
            placeholder_text="Ej: dame el reporte del cliente 42451020",
            height=42,
            font=ctk.CTkFont(size=14),
        )
        self.text_entry.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.text_entry.bind("<Return>", lambda e: self.on_send())

        self.send_btn = ctk.CTkButton(
            entry_row, text="Analizar", width=120, height=42, command=self.on_send
        )
        self.send_btn.grid(row=0, column=1)

        # Resultado
        result = ctk.CTkFrame(top, corner_radius=10)
        result.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 8))
        result.grid_columnconfigure((0, 1, 2), weight=1)

        self.intent_value = self._result_card(result, 0, "Intencion detectada", "-")
        self.confidence_value = self._result_card(result, 1, "Confianza", "-")
        self.param_value = self._result_card(result, 2, "Parametro detectado", "-")

        # Funcion / endpoint que se dispararia
        function_frame = ctk.CTkFrame(top, corner_radius=8, fg_color=("gray85", "gray17"))
        function_frame.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 16))
        ctk.CTkLabel(
            function_frame,
            text="Funcion / endpoint que se dispararia",
            font=ctk.CTkFont(size=12),
            text_color=("gray30", "gray60"),
        ).pack(anchor="w", padx=14, pady=(10, 0))
        self.function_value = ctk.CTkLabel(
            function_frame,
            text="-",
            font=ctk.CTkFont(size=15, weight="bold", family="Menlo"),
        )
        self.function_value.pack(anchor="w", padx=14, pady=(0, 12))

        # ---- Log panel ----
        log_frame = ctk.CTkFrame(self, corner_radius=12)
        log_frame.grid(row=2, column=0, sticky="nsew", padx=24, pady=(8, 24))
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(1, weight=1)

        log_header = ctk.CTkFrame(log_frame, fg_color="transparent")
        log_header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))
        log_header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            log_header, text="Registro (log)", font=ctk.CTkFont(size=15, weight="bold")
        ).grid(row=0, column=0, sticky="w")
        ctk.CTkButton(
            log_header, text="Limpiar", width=90, command=self.clear_log
        ).grid(row=0, column=1, sticky="e")

        self.log_box = ctk.CTkTextbox(
            log_frame, font=ctk.CTkFont(family="Menlo", size=12), wrap="word"
        )
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 16))
        self.log_box.configure(state="disabled")

    def _result_card(self, parent, col, title, value):
        card = ctk.CTkFrame(parent, corner_radius=8, fg_color=("gray85", "gray17"))
        card.grid(row=0, column=col, sticky="nsew", padx=8, pady=12)
        ctk.CTkLabel(
            card, text=title, font=ctk.CTkFont(size=12), text_color=("gray30", "gray60")
        ).pack(anchor="w", padx=14, pady=(10, 0))
        value_label = ctk.CTkLabel(
            card, text=value, font=ctk.CTkFont(size=18, weight="bold")
        )
        value_label.pack(anchor="w", padx=14, pady=(0, 12))
        return value_label

    # ---------- Logging ----------
    def enqueue_log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[{timestamp}] {message}")

    def _drain_log_queue(self):
        while not self.log_queue.empty():
            line = self.log_queue.get_nowait()
            self.log_box.configure(state="normal")
            self.log_box.insert("end", line + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        self.after(150, self._drain_log_queue)

    def clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    # ---------- Proyecto ----------
    def on_project_change(self, label):
        project = next(
            (p for p in PROJECTS if self._project_label(p) == label), None
        )
        if project is None or project == self.current_project:
            return
        self.current_project = project
        self.engine = NluEngine(project, log_callback=self.enqueue_log)
        self.status_label.configure(text="Estado: sin modelo cargado")
        self.intent_value.configure(text="-")
        self.confidence_value.configure(text="-")
        self.param_value.configure(text="-")
        self.function_value.configure(text="-")
        self.enqueue_log(f"Proyecto cambiado a: {self._project_label(project)}")
        self._refresh_pending_count()

    def _refresh_pending_count(self):
        count = revision.contar_pendientes(self.current_project)
        self.pending_btn.configure(text=f"Revisar pendientes ({count})")

    # ---------- Acciones ----------
    def _run_async(self, target, on_done=None):
        def wrapper():
            try:
                result = target()
                if on_done:
                    self.after(0, lambda: on_done(result))
            except Exception as exc:  # noqa: BLE001
                self.enqueue_log(f"ERROR: {exc}")
                self.enqueue_log(traceback.format_exc(limit=3))
                self.after(0, self._enable_buttons)

        self._disable_buttons()
        threading.Thread(target=wrapper, daemon=True).start()

    def _disable_buttons(self):
        for btn in (self.train_btn, self.load_btn, self.send_btn):
            btn.configure(state="disabled")

    def _enable_buttons(self):
        for btn in (self.train_btn, self.load_btn, self.send_btn):
            btn.configure(state="normal")

    def on_train(self):
        self.enqueue_log("Iniciando entrenamiento... esto puede tardar unos minutos.")
        self.status_label.configure(text="Estado: entrenando...")
        engine = self.engine

        def done(_):
            self.status_label.configure(text="Estado: modelo entrenado y cargado")
            self._enable_buttons()

        self._run_async(engine.train, on_done=done)

    def on_load(self):
        engine = self.engine

        def task():
            engine.ensure_loaded()
            return True

        def done(_):
            self.status_label.configure(text="Estado: modelo cargado")
            self._enable_buttons()

        self.enqueue_log("Cargando ultimo modelo entrenado...")
        self._run_async(task, on_done=done)

    def on_send(self):
        text = self.text_entry.get().strip()
        if not text:
            return
        self.enqueue_log(f'Consulta: "{text}"')
        engine = self.engine

        def task():
            return engine.parse(text)

        def done(result):
            self._show_result(result)
            self._enable_buttons()

        self._run_async(task, on_done=done)

    def _show_result(self, result):
        cfg = self._project_cfg()
        intent = result.get("intent", {}) or {}
        intent_name = intent.get("name") or "desconocida"
        confidence = intent.get("confidence") or 0.0
        entities = result.get("entities", []) or []
        text = result.get("text", "")

        label = cfg["intent_labels"].get(intent_name, intent_name)
        self.intent_value.configure(text=label)
        self.confidence_value.configure(text=f"{confidence * 100:.1f}%")

        value = None
        entity_type = None
        for etype in cfg["entity_types"]:
            found = [e["value"] for e in entities if e.get("entity") == etype]
            if found:
                entity_type, value = etype, found[0]
                break

        self.param_value.configure(text=f"{entity_type}: {value}" if value else "-")

        function_template = cfg["function_map"].get(intent_name)
        if function_template and value:
            function_call = function_template.format(value=value)
        elif function_template:
            function_call = f"{function_template} (sin parametro detectado)"
        else:
            function_call = "-"
        self.function_value.configure(text=function_call)

        self.enqueue_log(
            f"-> Intencion: {intent_name} ({confidence * 100:.1f}%) | "
            f"Parametro: {entity_type}={value} | Funcion: {function_call}"
        )

        if confidence < CONFIDENCE_THRESHOLD:
            revision.registrar_caso(
                self.current_project, text, intent_name, confidence, entities
            )
            self.enqueue_log(
                f"Confianza baja (<{CONFIDENCE_THRESHOLD * 100:.0f}%): caso guardado "
                f"para revision manual."
            )
            self._refresh_pending_count()

    # ---------- Revision de pendientes ----------
    def open_review_window(self):
        ReviewWindow(self)


class ReviewWindow(ctk.CTkToplevel):
    def __init__(self, app: "App"):
        super().__init__(app)
        self.app = app
        self.project = app.current_project
        self.cfg = app._project_cfg()
        self.title(f"Revisar pendientes - {app._project_label(self.project)}")
        self.geometry("640x420")

        self.casos = revision.listar_pendientes(self.project)
        self.idx = 0

        self.grid_columnconfigure(0, weight=1)

        self.counter_label = ctk.CTkLabel(self, font=ctk.CTkFont(size=13))
        self.counter_label.grid(row=0, column=0, sticky="w", padx=20, pady=(16, 4))

        self.text_box = ctk.CTkTextbox(self, height=70, font=ctk.CTkFont(size=14))
        self.text_box.grid(row=1, column=0, sticky="ew", padx=20, pady=(0, 8))
        self.text_box.configure(state="disabled")

        self.guess_label = ctk.CTkLabel(self, font=ctk.CTkFont(size=12), text_color=("gray30", "gray60"))
        self.guess_label.grid(row=2, column=0, sticky="w", padx=20, pady=(0, 12))

        form = ctk.CTkFrame(self, fg_color="transparent")
        form.grid(row=3, column=0, sticky="ew", padx=20)
        form.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(form, text="Intencion correcta:").grid(row=0, column=0, sticky="w", pady=6)
        intent_options = list(self.cfg["intent_labels"].keys()) + ["(ignorar - no es un caso valido)"]
        self.intent_menu = ctk.CTkOptionMenu(form, values=intent_options, width=280)
        self.intent_menu.grid(row=0, column=1, sticky="w", padx=(12, 0))

        ctk.CTkLabel(form, text="Tipo de parametro:").grid(row=1, column=0, sticky="w", pady=6)
        entity_options = list(self.cfg["entity_types"]) + ["(ninguno)"]
        self.entity_type_menu = ctk.CTkOptionMenu(form, values=entity_options, width=280)
        self.entity_type_menu.grid(row=1, column=1, sticky="w", padx=(12, 0))

        ctk.CTkLabel(form, text="Valor del parametro:").grid(row=2, column=0, sticky="w", pady=6)
        self.entity_value_entry = ctk.CTkEntry(form, width=280)
        self.entity_value_entry.grid(row=2, column=1, sticky="w", padx=(12, 0))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=4, column=0, sticky="ew", padx=20, pady=20)
        ctk.CTkButton(
            btn_row, text="Guardar y sumar a entrenamiento", command=self.on_guardar
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_row, text="Ignorar (no es valido)", fg_color="transparent",
            border_width=1, command=self.on_ignorar
        ).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            btn_row, text="Cerrar", fg_color="transparent", border_width=1,
            command=self.destroy
        ).pack(side="right")

        self._mostrar_actual()

    def _mostrar_actual(self):
        if self.idx >= len(self.casos):
            self.counter_label.configure(text="No hay mas casos pendientes.")
            self.text_box.configure(state="normal")
            self.text_box.delete("1.0", "end")
            self.text_box.insert("end", "(todo revisado)")
            self.text_box.configure(state="disabled")
            self.guess_label.configure(text="")
            for widget in (self.intent_menu, self.entity_type_menu, self.entity_value_entry):
                widget.configure(state="disabled")
            return

        caso = self.casos[self.idx]
        self.counter_label.configure(
            text=f"Caso {self.idx + 1} de {len(self.casos)}  (fecha: {caso['fecha']})"
        )
        self.text_box.configure(state="normal")
        self.text_box.delete("1.0", "end")
        self.text_box.insert("end", caso["texto"])
        self.text_box.configure(state="disabled")

        label = self.cfg["intent_labels"].get(caso["intent_predicho"], caso["intent_predicho"])
        self.guess_label.configure(
            text=f"El modelo dudo entre: {label} (confianza {caso['confianza'] * 100:.1f}%)"
        )

        self.intent_menu.set(caso["intent_predicho"] or "")
        entity_types = self.cfg["entity_types"]
        self.entity_type_menu.set(entity_types[0] if entity_types else "(ninguno)")
        self.entity_value_entry.delete(0, "end")

        entidades = json.loads(caso["entidades"] or "[]")
        if entidades:
            self.entity_type_menu.set(entidades[0].get("entity", entity_types[0] if entity_types else "(ninguno)"))
            self.entity_value_entry.insert(0, entidades[0].get("value", ""))

    def on_guardar(self):
        if self.idx >= len(self.casos):
            return
        caso = self.casos[self.idx]
        intent_elegido = self.intent_menu.get()

        if intent_elegido == "(ignorar - no es un caso valido)":
            self.on_ignorar()
            return

        entidad_tipo = self.entity_type_menu.get()
        if entidad_tipo == "(ninguno)":
            entidad_tipo = None
        entidad_valor = self.entity_value_entry.get().strip() or None

        try:
            frase = revision.agregar_ejemplo_a_nlu(
                self.project, intent_elegido, caso["texto"], entidad_tipo, entidad_valor
            )
            revision.marcar_revisado(
                self.project, caso["id"], intent_elegido, entidad_tipo, entidad_valor
            )
            self.app.enqueue_log(
                f"Ejemplo agregado a data/nlu.yml del intent '{intent_elegido}': \"{frase}\". "
                f"Reentrena el modelo para que lo aprenda."
            )
        except Exception as exc:  # noqa: BLE001
            self.app.enqueue_log(f"ERROR al guardar ejemplo: {exc}")

        self.idx += 1
        self.app._refresh_pending_count()
        self._mostrar_actual()

    def on_ignorar(self):
        if self.idx >= len(self.casos):
            return
        caso = self.casos[self.idx]
        revision.marcar_ignorado(self.project, caso["id"])
        self.app.enqueue_log(f'Caso ignorado (no se suma a entrenamiento): "{caso["texto"]}"')
        self.idx += 1
        self.app._refresh_pending_count()
        self._mostrar_actual()


if __name__ == "__main__":
    app = App()
    app.mainloop()
