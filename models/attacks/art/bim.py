import numpy as np
from dataclasses import dataclass
from art.attacks.evasion import BasicIterativeMethod
from utils.wrapper import create_art_classifier
from models.attacks.base_attack import BaseAttack

@dataclass
class BIMConfig:
    eps: float
    eps_step: float
    max_iter: int
    batch_size: int
    targeted: bool = False
    target_class: int = 0

class BIMAttack(BaseAttack):
    def __init__(self):
        super().__init__(name="BIM", tool="ART")

    def get_params(self, form) -> BIMConfig:
        try:
            targeted_str = form.get("bim_targeted") or "False"
            targeted = targeted_str.lower() == "true"
            target_class = int(form.get("bim_target_class") or 0)
            return BIMConfig(
                eps=float(form.get("bim_eps") or 0.3),
                eps_step=float(form.get("bim_eps_step") or 0.01),
                max_iter=int(form.get("bim_max_iter") or 10),
                batch_size=int(form.get("bim_batch_size") or 32),
                targeted=targeted,
                target_class=target_class
            )
        except ValueError:
            raise ValueError("Invalid BIM Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: BIMConfig, **kwargs):
        try:
            art_classifier = create_art_classifier(model, X_test, y_test, attack_type="BIM")

            attack = BasicIterativeMethod(
                estimator=art_classifier,
                eps=config.eps,
                eps_step=config.eps_step,
                max_iter=config.max_iter,
                batch_size=config.batch_size,
                targeted=config.targeted,
                verbose=True
            )

            if config.targeted:
                if len(y_test.shape) == 2:
                    y_target = np.zeros_like(y_test)
                    y_target[:, config.target_class] = 1
                else:
                    y_target = np.full_like(y_test, config.target_class)

                X_adv = attack.generate(x=X_test, y=y_target)
            else:
                X_adv = attack.generate(x=X_test)
            return X_adv


        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
