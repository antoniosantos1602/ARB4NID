import numpy as np
from dataclasses import dataclass
from art.attacks.evasion import FastGradientMethod
from utils.wrapper import create_art_classifier
from models.attacks.base_attack import BaseAttack

@dataclass
class FGMConfig:
    eps: float
    eps_step: float
    batch_size: int
    norm: any
    targeted: bool
    target_class: int

class FGMAttack(BaseAttack):
    def __init__(self):
        super().__init__(name="FGM", tool="ART")

    def get_params(self, form) -> FGMConfig:
        try:
            eps_raw = form.get("fgm_eps_manual") or form.get("fgm_eps")
            eps_step_raw = form.get("fgm_eps_step_manual") or form.get("fgm_eps_step")
            bs_raw = form.get("fgm_batch_size_manual") or form.get("fgm_batch_size")
            norm_val = form.get("fgm_norm") or "inf"
            target_class_raw = form.get("fgm_target_class", 0)

            if norm_val == "inf":
                art_norm = "inf"
            else:
                try:
                    art_norm = int(norm_val)
                except:
                    art_norm = "inf"

            return FGMConfig(
                eps=float(eps_raw or 0.3),
                eps_step=float(eps_step_raw or 0.1),
                batch_size=int(bs_raw or 32),
                norm=art_norm,
                targeted=form.get("fgm_targeted") == "True",
                target_class=int(target_class_raw)
            )
        except ValueError:
            raise ValueError(f"Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: FGMConfig, **kwargs):
        try:
            art_classifier = create_art_classifier(model, X_test, y_test, attack_type="FGM")

            attack = FastGradientMethod(
                estimator=art_classifier,
                norm=config.norm,
                eps=config.eps,
                eps_step=config.eps_step,
                batch_size=config.batch_size,
                targeted=config.targeted,
                minimal=True,
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
