from flask import Flask, current_app

from models.intelligence_cache import IntelligenceCache
from models.results_repository import ResultsRepository


def register_services(app: Flask) -> None:
    app.extensions["intelligence_cache"] = IntelligenceCache()
    app.extensions["results_repository"] = ResultsRepository()


def get_intelligence_cache() -> IntelligenceCache:
    return current_app.extensions["intelligence_cache"]


def get_results_repository() -> ResultsRepository:
    return current_app.extensions["results_repository"]
