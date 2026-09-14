import numpy as np
from dataclasses import dataclass
from art.attacks.evasion import BoundaryAttack as ARTBoundaryAttack
from utils.wrapper import create_art_classifier
from models.attacks.base_attack import BaseAttack

@dataclass
@dataclass
class BoundaryConfig:
    max_iter: int
    batch_size: int
    num_trial: int
    sample_size: int
    init_size: int
    targeted: bool
    target_class: int

class BoundaryAttackClass(BaseAttack):
    def __init__(self):
        super().__init__(name="BoundaryAttack", tool="ART")

    def get_params(self, form) -> BoundaryConfig:
        try:
            raw_targeted = form.get("boundary_targeted", "").strip().lower()
            targeted = raw_targeted in ("true", "1", "on", "yes")

            return BoundaryConfig(
                max_iter=int(form.get("boundary_max_iter") or 20),
                batch_size=int(form.get("boundary_batch_size") or 32),
                num_trial=int(form.get("boundary_num_trial") or 5),
                sample_size=int(form.get("boundary_sample_size") or 5),
                init_size=int(form.get("boundary_init_size") or 10),
                targeted=targeted,
                target_class=int(form.get("boundary_target_class") or 0)
            )
        except ValueError:
            raise ValueError("Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: BoundaryConfig, **kwargs):
        try:
            art_classifier = create_art_classifier(model, X_test, y_test, attack_type="BoundaryAttack")

            attack = ARTBoundaryAttack(
                estimator=art_classifier,
                targeted=config.targeted,
                max_iter=config.max_iter,
                num_trial=config.num_trial,
                sample_size=config.sample_size,
                init_size=config.init_size,
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
