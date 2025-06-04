# app/art_attacks_evasion/zoo.py
import numpy as np
from art.attacks.evasion import ZooAttack
from app.utils.wrapper import create_art_classifier
from app.utils.param_utils import get_param
import time

def run_zoo(model, X, y, **params):
    """
    Run ZOO attack.
    """
    max_iter = get_param(params, "max_iter", default=10, expected_type=int, min_val=1, max_val=1000)
    learning_rate = get_param(params, "learning_rate", default=1e-1, expected_type=float)
    binary_search_steps = get_param(params, "binary_search_steps", default=1, expected_type=int)
    initial_const = get_param(params, "initial_const", default=1e-3, expected_type=float)
    batch_size = get_param(params, "batch_size", default=1, expected_type=int, min_val=32, max_val=1024)

    classifier = create_art_classifier(model, X, y, attack_type='ZOO')

    start = time.time()

    attack = ZooAttack(
        classifier=classifier,
        max_iter=max_iter,
        learning_rate=learning_rate,
        binary_search_steps=binary_search_steps,
        initial_const=initial_const,
        batch_size=batch_size,
        nb_parallel=5
    )

    X_adv = attack.generate(X)
    y_pred = classifier.predict(X)
    y_pred_adv = classifier.predict(X_adv)

    end = time.time()
    exec_time = end - start

    fooled_idx = np.argmax(y_pred, axis=1) != np.argmax(y_pred_adv, axis=1)
    asr = np.sum(fooled_idx) / len(y)
    robust_accuracy = 1 - asr

    return X_adv, fooled_idx, robust_accuracy, asr, exec_time
