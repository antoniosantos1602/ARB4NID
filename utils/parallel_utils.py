import logging
import os

from werkzeug.datastructures import MultiDict

from models.attacks.art.attack_router import run_attack
from models.attacks.foolbox.foolbox_router import run_foolbox_attack_router

logger = logging.getLogger(__name__)


class DummyRequest:
    def __init__(self, form):
        self.form = MultiDict(form) if isinstance(form, dict) else MultiDict(form)


def run_single_attack(args):
    attack, model_path, shared_params = args
    logger.info("[ART] PID: %s - Executing: %s", os.getpid(), attack)
    clean_attack = attack.replace("_ART", "").strip()
    dummy_request = DummyRequest(shared_params["form"])
    return run_attack(
        clean_attack,
        dummy_request,
        [model_path],
        shared_params["X_test"],
        shared_params["y_test"],
        shared_params["df_original"],
        shared_params["column_names"],
        intelligence=shared_params["intelligence"],
    )


def run_single_foolbox_attack(args):
    attack_type, model_paths, shared_params = args
    logger.info("[FOOLBOX] PID: %s - Executing: %s", os.getpid(), attack_type)
    clean_attack = attack_type.replace("_FOOLBOX", "").strip()
    form_data = dict(shared_params["form"])
    form_data["attack_name"] = clean_attack
    dummy_request = DummyRequest(form_data)
    return run_foolbox_attack_router(
        [clean_attack],
        dummy_request,
        model_paths,
        shared_params["X_test"],
        shared_params["y_test"],
        shared_params["df_original"],
        shared_params["column_names"],
        intelligence=shared_params["intelligence"],
    )


def run_single_a2pm_attack(args):
    model_paths, shared_params = args
    logger.info("[A2PM] PID: %s - Executing A2PM", os.getpid())
    dummy_request = DummyRequest(shared_params["form"])
    return run_attack(
        "A2PM",
        dummy_request,
        model_paths,
        shared_params["X_test"],
        shared_params["y_test"],
        shared_params["df_original"],
        shared_params["column_names"],
        intelligence=shared_params["intelligence"],
    )
