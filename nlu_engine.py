"""
Wrapper liviano alrededor de Rasa para entrenar y consultar modelos NLU
desde la interfaz grafica. Cada "proyecto" es una carpeta independiente
dentro de proyectos/ con su propio nlu.yml, config.yml, domain.yml y
carpeta de modelos entrenados, para poder tener varios bots (distintas
empresas/casos de uso) sin que se pisen entre si.
"""
import asyncio
import glob
import logging
import os

import structlog

os.environ.setdefault("LOG_LEVEL", "ERROR")
os.environ.setdefault("SANIC_ACCESS_LOG", "False")
logging.getLogger("rasa").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR)
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, "proyectos")


def list_projects():
    if not os.path.isdir(PROJECTS_DIR):
        return []
    return sorted(
        name
        for name in os.listdir(PROJECTS_DIR)
        if os.path.isdir(os.path.join(PROJECTS_DIR, name))
        and not name.startswith(".")
    )


class NluEngine:
    def __init__(self, project, log_callback=None):
        self.project = project
        self.project_dir = os.path.join(PROJECTS_DIR, project)
        self.config_path = os.path.join(self.project_dir, "config.yml")
        self.nlu_data_path = os.path.join(self.project_dir, "data", "nlu.yml")
        self.models_dir = os.path.join(self.project_dir, "models")

        self.agent = None
        self.model_path = None
        self.log = log_callback or (lambda msg: None)

    def latest_model_path(self):
        os.makedirs(self.models_dir, exist_ok=True)
        candidates = sorted(
            glob.glob(os.path.join(self.models_dir, "*.tar.gz")),
            key=os.path.getmtime,
        )
        return candidates[-1] if candidates else None

    def train(self):
        from rasa.model_training import train_nlu

        self.log(f"[{self.project}] Entrenando modelo NLU con data/nlu.yml ...")
        os.makedirs(self.models_dir, exist_ok=True)
        model_path = train_nlu(
            config=self.config_path,
            nlu_data=self.nlu_data_path,
            output=self.models_dir,
        )
        self.model_path = model_path
        self.log(f"[{self.project}] Modelo entrenado: {os.path.basename(model_path)}")
        self._load(model_path)
        return model_path

    def ensure_loaded(self):
        if self.agent is not None:
            return
        model_path = self.latest_model_path()
        if not model_path:
            raise RuntimeError(
                f"No hay ningun modelo entrenado todavia para '{self.project}'."
            )
        self._load(model_path)

    def _load(self, model_path):
        from rasa.core.agent import Agent

        self.log(f"[{self.project}] Cargando modelo {os.path.basename(model_path)} ...")
        self.agent = Agent.load(model_path)
        self.model_path = model_path
        self.log(f"[{self.project}] Modelo cargado y listo para recibir consultas.")

    def parse(self, text):
        self.ensure_loaded()
        result = asyncio.run(self.agent.parse_message(text))
        return result
