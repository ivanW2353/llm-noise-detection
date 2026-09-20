from dataclasses import dataclass
from typing import Any, Protocol, Iterable

@dataclass
class Sample:
    id: str
    messages: list[dict[str,str]]
    noise: str='none'
    meta: dict[str,Any]|None=None

class Provider(Protocol):
    def rows(self) -> Iterable[Sample]: ...
