import numpy as np
from dataclasses import dataclass
from art.attacks.evasion import SaliencyMapMethod
from utils.wrapper import create_art_classifier
from models.attacks.base_attack import BaseAttack

@dataclass
class JSMAConfig:
    theta: float
    gamma: float
    batch_size: int
    targeted: bool
    target_class : int

class JSMAAttack(BaseAttack):
    def __init__(self):
        super().__init__(name="JSMA", tool="ART")

    def get_params(self, form) -> JSMAConfig:
        try:
            theta_raw = form.get("jsma_theta_manual") or form.get("jsma_theta")
            gamma_raw = form.get("jsma_gamma_manual") or form.get("jsma_gamma")
            bs_raw = form.get("jsma_batch_size_manual") or form.get("jsma_batch_size")
            target_class_raw = form.get("jsma_target_class" , 0)

            return JSMAConfig(
                theta=float(theta_raw or 0.1),
                gamma=float(gamma_raw or 0.1),
                batch_size=int(bs_raw or 32),
                targeted=form.get("jsma_targeted") == "True",
                target_class=int(target_class_raw),
            )
        except ValueError as e:
            raise ValueError(f"Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: JSMAConfig, **kwargs):
        try:
            art_classifier = create_art_classifier(model, X_test, y_test, attack_type="JSMA")

            attack = SaliencyMapMethod(
                classifier=art_classifier,
                theta=config.theta,
                gamma=config.gamma,
                batch_size=config.batch_size,
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
