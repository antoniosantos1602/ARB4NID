import numpy as np
import pandas as pd
from dataclasses import dataclass
from art.attacks.evasion import ZooAttack
from art.utils import to_categorical
from utils.wrapper import create_art_classifier
from models.attacks.base_attack import BaseAttack

@dataclass
class ZOOConfig:
    max_iter: int
    binary_search_steps: int
    confidence: float
    learning_rate: float
    targeted: bool
    target_class: int

class ZOOAttack(BaseAttack):
    def __init__(self):
        super().__init__(name="ZOO", tool="ART")

    def get_params(self, form) -> ZOOConfig:
        try:
            return ZOOConfig(
                max_iter=int(form.get("zoo_max_iter") or 10),
                binary_search_steps=int(form.get("zoo_binary_search_steps") or 1),
                confidence=float(form.get("zoo_confidence") or 0.0),
                learning_rate=float(form.get("zoo_learning_rate") or 0.5),
                targeted=form.get("zoo_targeted") == "True",
                target_class=int(form.get("zoo_target_class") or 0)
            )
        except ValueError:
            raise ValueError("Invalid ZOO Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: ZOOConfig, **kwargs):
        try:
            X_input = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            y_input = y_test.values if isinstance(y_test, pd.Series) else y_test

            art_classifier = create_art_classifier(model, X_input, y_input, attack_type="ZOO")

            attack = ZooAttack(
                classifier=art_classifier,
                confidence=config.confidence,
                targeted=config.targeted,
                learning_rate=config.learning_rate,
                max_iter=config.max_iter,
                binary_search_steps=config.binary_search_steps,
                initial_const=1e-3,
                abort_early=True,
                use_resize=False,
                use_importance=True,
                nb_parallel=1,
                batch_size=1,
                variable_h=0.0001,
            )

            if config.targeted:
                attack_mask = (y_input != config.target_class)
                X_to_attack = X_input[attack_mask]

                if len(X_to_attack) == 0:
                    return X_input

                n_classes = art_classifier.nb_classes
                y_target_oh = to_categorical(np.full(len(X_to_attack), config.target_class), nb_classes=n_classes)

                X_adv_partial = attack.generate(x=X_to_attack, y=y_target_oh)

                X_adv = X_input.copy()
                X_adv[attack_mask] = X_adv_partial
            else:
                X_adv = attack.generate(x=X_input)

            return X_adv
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
