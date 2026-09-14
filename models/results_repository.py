from typing import Optional


class ResultsRepository:

    def __init__(self) -> None:
        self._store: dict = {}

    def add(self, result_dict: dict) -> str:
        attack_id = f"{result_dict['attack_name']}_{result_dict['model_name']}"
        self._store[attack_id] = result_dict
        return attack_id

    def get(self, attack_id: str) -> Optional[dict]:
        return self._store.get(attack_id)

    def delete(self, attack_id: str) -> Optional[dict]:
        return self._store.pop(attack_id, None)

    def all(self) -> list:
        return [
            {"attack_id": k, **v}
            for k, v in self._store.items()
        ]

    def clear(self) -> bool:
        self._store = {}
        return True
