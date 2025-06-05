from flask import Blueprint, request, jsonify
import os
import numpy as np

from sklearn.metrics import accuracy_score
from app.utils.utils import load_dataset, load_model, predict_model
from app.utils.attack_runner import safe_run_attack
from app.art_attacks_evasion.fgsm import run_fgm
from app.art_attacks_evasion.jsma import run_jsma
from app.art_attacks_evasion.zoo import run_zoo

evasion = Blueprint('evasion', __name__)

# -------------------------
# Função para executar o ataque
# -------------------------
def execute_attack(name, model, X, y, params):
    if name == 'FGSM':
        return safe_run_attack(run_fgm, model, X, y, params, name)
    elif name == 'JSMA':
        return safe_run_attack(run_jsma, model, X, y, params, name)
    elif name == 'ZOO':
        return safe_run_attack(run_zoo, model, X, y, params, name)
    else:
        return None

# -------------------------
# Cálculo do RSD por classe
#[0,1,1,0,0]
# EX: enganou 2 da classe 0
# ex: enganou 1 da classe 1
# RA0= 2/3 = 0.66
# RA1= 1/3 = 0.33
#Mean= (0.66+0.33)/2 = 0.495
#desvio padrao = sqrt(((0.66-0.495)^2 + (0.33-0.495)^2))
#
# -------------------------

def calculate_rsd_by_class(y_initial, y_survived):
    classes = np.unique(y_initial)
    robust_accuracies = []

    for cls in classes:
        total_class = np.sum(y_initial == cls)
        survived_class = np.sum(y_survived == cls)

        if total_class > 0:
            ra = survived_class / total_class
            robust_accuracies.append(ra)
            print(f"Classe {cls} - Robust Accuracy: {ra:.2f}%")

    if robust_accuracies:
        mean_ra = np.mean(robust_accuracies)
        print(mean_ra)
        std_ra = np.std(robust_accuracies)
        rsd_ra = (std_ra / mean_ra)*100 if mean_ra > 0 else 0.0
    else:
        rsd_ra = 0.0

    return rsd_ra

# -------------------------
# Route
# -------------------------
@evasion.route('/run_evasion', methods=['POST'])
def run_evasion_attacks():
    data = request.get_json()
    ordered_attacks = data.get("attacks", [])

    # Carregar dados e modelo
    X_train, y_train, X_test, y_test = load_dataset("uploads/train.csv", "uploads/test.csv")
    model, scaler = load_model("uploads/model.joblib")

    if scaler:
        X_test = scaler.transform(X_test)

    y_test_initial = y_test.copy()
    initial_total = len(y_test)

    y_pred_clean = predict_model(model, X_test)
    clean_accuracy = accuracy_score(y_test, y_pred_clean)

    results = []

    for attack in ordered_attacks:
        name, params = attack['name'], attack['params']

        # Conjunto antes do ataque (para cálculo do RSD isolado)
        X_before = X_test.copy()
        y_before = y_test.copy()

        result = execute_attack(name, model, X_before, y_before, params)

        if result is None:
            continue

        X_adv, fooled_idx, robust_acc, asr, exec_time = result

        # Atualiza o conjunto para o próximo ataque
        mask = ~fooled_idx
        X_test = X_before[mask]
        y_test = y_before[mask]

        # Cálculo do RSD isolado: antes vs depois do ataque atual
        rsd_ra = calculate_rsd_by_class(y_before, y_test)

        results.append({
            "attack": name,
            "result": {
                "fooled": int(np.sum(fooled_idx)),
                "remaining": int(len(y_test)),
                "robust_accuracy": round(robust_acc * 100, 2),
                "asr": round(asr * 100, 2),
                "execution_time": round(exec_time, 2),
                "rsd": round(rsd_ra, 4)
            }
        })

    # Métricas finais (efeito acumulado de todos os ataques)
    total_time = sum(r["result"]["execution_time"] for r in results if "execution_time" in r["result"])
    final_rsd = calculate_rsd_by_class(y_test_initial, y_test)
    final_robust_accuracy = len(y_test) / initial_total if initial_total > 0 else 0.0
    final_asr = 1 - final_robust_accuracy

    results.append({
        "attack": "Final Summary",
        "result": {
            "initial": initial_total,
            "remaining": len(y_test),
            "final_robust_accuracy": round(final_robust_accuracy * 100, 2),
            "final_asr": round(final_asr * 100, 2),
            "total_time": round(total_time, 2),
            "rsd": round(final_rsd, 4)
        }
    })

    results.insert(0, {
    "attack": "Clean Accuracy",
    "result": {
        "accuracy": round(clean_accuracy * 100, 2)
    }
    })

    return jsonify(results)
