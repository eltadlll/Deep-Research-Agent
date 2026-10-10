from __future__ import annotations

from ..config import Config
from ..llm import LLM
from ..state import ResearchState


class BaseAgent:
    name = "Agent"
    role = ""

    def __init__(self, llm: LLM, cfg: Config, state: ResearchState, toolkit=None):
        self.llm = llm
        self.cfg = cfg
        self.state = state
        self.tk = toolkit

    def log(self, msg: str) -> None:
        self.state.log(self.name, msg)

    @property
    def system(self) -> str:
        return (f"You are the {self.name} in a multi-agent research team. {self.role} "
                "Be accurate and concrete. Never invent facts, numbers, or sources.")
