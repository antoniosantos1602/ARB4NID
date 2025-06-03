from flask import Blueprint, request, jsonify
import os
from app.utils.utils import load_dataset, load_model
from app.utils.attack_runner import safe_run_attack
from app.art_attacks_evasion.fgsm import run_fgm
from app.art_attacks_evasion.jsma import run_jsma
import numpy as np

evasion = Blueprint('evasion', __name__)

@evasion.route('/run_evasion', methods=['POST'])
def run_evasion_attacks():
    data = request.get_json()
    ordered_attacks = data.get("attacks", [])

    # Load
    X_train, y_train, X_test, y_test = load_dataset("uploads/train.csv", "uploads/test.csv")
    model, scaler = load_model("uploads/model.joblib")

    if scaler:
        X_test = scaler.transform(X_test)

    results = []
    for attack in ordered_attacks:
        name, params = attack['name'], attack['params']
        if name == 'FGSM':
            result = safe_run_attack(run_fgm, model, X_test, y_test, params, name)
        elif name == 'JSMA':
            result = safe_run_attack(run_jsma, model, X_test, y_test, params, name)
        else:
            continue

        if result is None:
            continue

        X_adv, fooled_idx, robust_acc, asr, exec_time = result

        # Atualizar conjunto
        mask = ~fooled_idx
        X_test, y_test = X_test[mask], y_test[mask]

        results.append({
            "attack": name,
            "result": {
                "fooled": int(np.sum(fooled_idx)),
                "remaining": int(len(y_test)),
                "robust_accuracy": round(robust_acc * 100, 2),
                "asr": round(asr * 100, 2),
                "execution_time": round(exec_time, 2)
            }
        })

    return jsonify(results)
