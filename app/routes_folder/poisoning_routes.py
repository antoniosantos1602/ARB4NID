from flask import Blueprint, request, jsonify
from app.utils.utils import load_dataset, load_model, predict_model
from app.art_adversarial_training.adversarial_training import run_adversarial_training
from app.utils.wrapper import create_art_classifier
from art.attacks.evasion import FastGradientMethod
from sklearn.metrics import accuracy_score

import numpy as np

poisoning = Blueprint('poisoning', __name__)

@poisoning.route('/run_poisoning', methods=['POST'])
def run_poisoning_attacks():
    data = request.get_json()
    ordered_attacks = data.get("attacks", [])

    X_train, y_train, X_test, y_test = load_dataset("uploads/train.csv", "uploads/test.csv")
    model, scaler = load_model("uploads/model.joblib")


    if scaler:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)


    # Fazer predição com função genérica
    y_pred = predict_model(model, X_test)
    acc = accuracy_score(y_test, y_pred)

    results = []
    for attack in ordered_attacks:
        name, params = attack['name'], attack['params']

        if name in ['FGSM', 'PGD']:
            classifier, adv_attack = run_adversarial_training(model, X_train, y_train, name, **params)

            # Accuracy After Adversarial Training
            preds = classifier.predict(X_test)
            preds = np.argmax(preds, axis=1)
            acc_after_adv = accuracy_score(y_test, preds)

            # Robust accuracy
            #X_test_adv = adv_attack.generate(X_test)
            #adv_preds = classifier.predict(X_test_adv)
            #adv_preds = np.argmax(adv_preds, axis=1)
            #robust_acc = accuracy_score(y_test, adv_preds)

            results.append({
                "attack": name,
                "result": {
                    "accuracy_original_samples": round(acc * 100, 2),
                    "accuracy_after_training": round(acc_after_adv * 100, 2),

                }
            })
        else:
            results.append({
                "attack": name,
                "result": "Ataque não suportado para adversarial training."
            })

    return jsonify(results)
