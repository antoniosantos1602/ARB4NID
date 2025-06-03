import numpy as np
from art.attacks.evasion import SaliencyMapMethod
from app.utils.wrapper import create_art_classifier
from app.utils.param_utils import get_param
import time

def run_jsma(model, X, y, **params):
    """
    """
    #params
    theta = get_param(params, "theta", default=0.1, expected_type=float, min_val=0.0, max_val=1.0)
    gamma = get_param(params, "gamma", default=0.9, expected_type=float, min_val=0.0, max_val=1.0)
    batch_size = get_param(params, "batch_size", default=64, expected_type=int, min_val=32, max_val=1024)

    classifier = create_art_classifier(model, X, y, attack_type='JSMA')

    start = time.time()

    X_adv = SaliencyMapMethod(classifier=classifier,
                              theta=theta,
                              gamma=gamma,
                              batch_size=batch_size).generate(X)

    y_pred = classifier.predict(X)
    y_pred_adv = classifier.predict(X_adv)

    end = time.time()
    exec_time = end - start

    fooled_idx = np.argmax(y_pred, axis=1) != np.argmax(y_pred_adv, axis=1)

    asr = np.sum(fooled_idx) / len(y)
    robust_accuracy = 1 - asr


    return X_adv, fooled_idx, robust_accuracy, asr, exec_time
