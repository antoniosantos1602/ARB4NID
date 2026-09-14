import numpy as np
from dataclasses import dataclass
from art.attacks.evasion import ProjectedGradientDescent
from utils.wrapper import create_art_classifier
from models.attacks.base_attack import BaseAttack

@dataclass
class PGDConfig:
    eps: float
    eps_step: float
    max_iter: int
    batch_size: int
    num_random_init: int
    norm: any
    targeted: bool
    target_class: int

class PGDAttack(BaseAttack):
    def __init__(self):
        super().__init__(name="PGD", tool="ART")

    def get_params(self, form) -> PGDConfig:
        try:
            norm_val = form.get("pgd_norm") or "inf"

            if norm_val == "inf":
                art_norm = "inf"
            else:
                try:
                    art_norm = int(norm_val)
                except:
                    art_norm = "inf"

            return PGDConfig(
                eps=float(form.get("pgd_eps") or 0.5),
                eps_step=float(form.get("pgd_eps_step") or 0.1),
                max_iter=int(form.get("pgd_max_iter") or 10),
                batch_size=int(form.get("pgd_batch_size") or 32),
                num_random_init=int(form.get("num_random_init_pgd") or 0),
                norm=art_norm,
                targeted=form.get("pgd_targeted") == "True",
                target_class=int(form.get("pgd_target_class") or 0)
            )
        except ValueError:
            raise ValueError("Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: PGDConfig, **kwargs):
        try:
            art_classifier = create_art_classifier(model, X_test, y_test, attack_type="PGD")

            attack = ProjectedGradientDescent(
                estimator=art_classifier,
                norm=config.norm,
                eps=config.eps,
                eps_step=config.eps_step,
                max_iter=config.max_iter,
                targeted=config.targeted,
                batch_size=config.batch_size,
                num_random_init=config.num_random_init,
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
