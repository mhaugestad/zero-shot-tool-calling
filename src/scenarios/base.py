from typing import Protocol
from src.domain import ToolSelectionExample

class Scenario(Protocol):
    name: str

    def generate(self, example: ToolSelectionExample) -> ToolSelectionExample:
        ...