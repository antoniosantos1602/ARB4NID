import numpy as np
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.defences.trainer import AdversarialTrainer
from app.utils.wrapper import create_art_classifier
from app.utils.param_utils import get_param

def get_attack(attack_name, classifier, params):
    epsilon = get_param(params, "epsilon", default=0.2, expected_type=float)
    eps_step = get_param(params, "eps_step", default=0.04, expected_type=float)
    batch_size = get_param(params, "batch_size", default=64, expected_type=int)
    norm_raw = params.get("norm", 2)
    norm = np.inf if norm_raw == "inf" else get_param(params, "norm", default=2, expected_type=int)

    if attack_name == "FGSM":
        return FastGradientMethod(estimator=classifier, eps=epsilon, eps_step=eps_step, batch_size=batch_size, norm=norm)
    elif attack_name == "PGD":
        return ProjectedGradientDescent(estimator=classifier, eps=epsilon, eps_step=eps_step, batch_size=batch_size, norm=norm)
    else:
        raise ValueError(f"Ataque '{attack_name}' não suportado para adversarial training.")

def run_adversarial_training(model, X_train, y_train, attack_name, **params):
    X_train = np.array(X_train).astype(np.float32)
    y_train = np.array(y_train).astype(int)

    batch_size = get_param(params, "batch_size", default=64, expected_type=int)
    nb_epochs = get_param(params, "nb_epochs", default=5, expected_type=int)

    classifier = create_art_classifier(model, X_train, y_train, attack_type=attack_name)
    attack = get_attack(attack_name, classifier, params)

    trainer = AdversarialTrainer(classifier, attack)
    trainer.fit(X_train, y_train, nb_epochs=nb_epochs, batch_size=batch_size)

    return classifier, attack  # Devolve o modelo treinado + o ataque (para robust acc)
