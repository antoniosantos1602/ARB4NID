import numpy as np

def safe_run_attack(attack_fn, model, X, y, params, attack_name):
    """
    Executa um ataque adversarial apenas se houver >= 2 classes em y.
    """
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        print(f"[IGNORADO] {attack_name}: só {len(unique_classes)} classe(s) restante(s): {unique_classes}")
        return None

    return attack_fn(model, X, y, **params)
