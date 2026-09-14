import numpy as np
from dataclasses import dataclass
from art.attacks.evasion import HopSkipJump as ARTHopSkipJump
from utils.wrapper import create_art_classifier
from models.attacks.base_attack import BaseAttack

@dataclass
class HSJConfig:
    max_iter: int
    max_eval: int
    init_eval: int
    init_size: int
    batch_size: int
    norm: any
    targeted: bool
    target_class: int

class HopSkipJumpAttack(BaseAttack):
    def __init__(self):
        super().__init__(name="HopSkipJump", tool="ART")

    def get_params(self, form) -> HSJConfig:
        try:
            raw_targeted = form.get("hsj_targeted", "").strip().lower()
            targeted = raw_targeted in ("true", "1", "on", "yes")

            norm_val = form.get("hsj_norm") or form.get("norm")
            actual_norm = 2 if norm_val == "2" else np.inf

            max_eval = int(form.get("max_eval") or 100)
            init_eval = int(form.get("init_eval") or 20)
            if init_eval >= max_eval:
                init_eval = max(1, max_eval // 5)

            return HSJConfig(
                max_iter=int(form.get("adv_max_iter") or 10),
                max_eval=max_eval,
                init_eval=init_eval,
                init_size=int(form.get("init_size") or 10),
                batch_size=int(form.get("hsj_batch_size") or 32),
                norm=actual_norm,
                targeted=targeted,
                target_class=int(form.get("hsj_target_class") or 0)
            )
        except ValueError:
            raise ValueError("Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: HSJConfig, **kwargs):
        try:
            art_classifier = create_art_classifier(model, X_test, y_test, attack_type="HopSkipJump")

            attack = ARTHopSkipJump(
                classifier=art_classifier,
                batch_size=config.batch_size,
                init_eval=config.init_eval,
                init_size=config.init_size,
                max_eval=config.max_eval,
                max_iter=config.max_iter,
                targeted=config.targeted,
                norm=config.norm,
                verbose=True
            )

            if config.targeted:
                if len(y_test.shape) == 2:
                    y_target = np.zeros_like(y_test)
                    y_target[:, config.target_class] = 1
                else:
                    y_target = np.full((len(y_test),), config.target_class)

                X_adv = attack.generate(x=X_test, y=y_target)
            else:
                X_adv = attack.generate(x=X_test)

            return X_adv
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
