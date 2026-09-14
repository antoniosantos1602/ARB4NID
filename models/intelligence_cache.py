from typing import Optional


class IntelligenceCache:

    def __init__(self) -> None:
        self._intelligence = None

    def save(self, obj) -> None:
        self._intelligence = obj

    def get(self) -> Optional[object]:
        return self._intelligence

    def clear(self) -> None:
        self._intelligence = None
