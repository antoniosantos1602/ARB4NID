from dataclasses import dataclass
import foolbox as fb
from models.attacks.base_attack import BaseAttack
from models.attacks.foolbox.attacks_foolbox import run_fb_logic

@dataclass
class FGMConfigFB:
    eps: float
    targeted: bool = False

class FGMFoolbox(BaseAttack):
    def __init__(self):
        super().__init__(name="FGM", tool="FOOLBOX")

    def get_params(self, form) -> FGMConfigFB:
        try:
            return FGMConfigFB(
                eps=float(form.get("fgm_fb_eps") or 0.2),
                targeted=False,
            )
        except ValueError:
            raise ValueError("Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: FGMConfigFB, **kwargs):
        try:
            return run_fb_logic(model, X_test, y_test, config, fb.attacks.L2FastGradientAttack)
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
