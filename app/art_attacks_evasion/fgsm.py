import numpy as np
from art.attacks.evasion import FastGradientMethod
from app.utils.wrapper import create_art_classifier
from app.utils.param_utils import get_param
import time

def run_fgm(model, X, y, **params):

    #params
    epsilon = get_param(params, "epsilon", default=0.1, expected_type=float, min_val=0.0, max_val=1.0)
    eps_step = get_param(params, "eps_step", default=0.02, expected_type=float, min_val=0.0, max_val=1.0)
    batch_size = get_param(params, "batch_size", default=64, expected_type=int, min_val=32, max_val=1024)

    norm_raw = params.get("norm", 2)
    if norm_raw == "inf":
        norm = np.inf
    else:
        norm = get_param(params, "norm", default=2, expected_type=int, min_val=1, max_val=2)


    #wrapper
    classifier = create_art_classifier(model, X, y, attack_type='FGM')

    start = time.time()

    X_adv = FastGradientMethod(
        estimator=classifier,
        eps=epsilon,
        eps_step=eps_step,
        batch_size=batch_size,
        norm=norm).generate(X)

    y_pred = classifier.predict(X)
    y_pred_adv = classifier.predict(X_adv)

    end = time.time()
    exec_time = end - start

    fooled_idx = np.argmax(y_pred, axis=1) != np.argmax(y_pred_adv, axis=1)

    asr = np.sum(fooled_idx) / len(y)
    robust_accuracy = 1 - asr


    return X_adv, fooled_idx, robust_accuracy, asr, exec_time
