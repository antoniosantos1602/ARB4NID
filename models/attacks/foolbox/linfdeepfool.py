from dataclasses import dataclass
import foolbox as fb
from models.attacks.base_attack import BaseAttack
from models.attacks.foolbox.attacks_foolbox import run_fb_logic

@dataclass
class LinfDeepFoolConfigFB:
    steps: int
    overshoot: float
    targeted: bool = False

class LinfDeepFoolFoolbox(BaseAttack):
    def __init__(self):
        super().__init__(name="LinfDeepFool", tool="FOOLBOX")

    def get_params(self, form) -> LinfDeepFoolConfigFB:
        try:
            return LinfDeepFoolConfigFB(
                steps=int(form.get("steps") or 50),
                overshoot=float(form.get("overshoot") or 0.02),
                targeted=False,
                target_class=0
            )
        except ValueError:
            raise ValueError("Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: LinfDeepFoolConfigFB, **kwargs):
        try:
            return run_fb_logic(model, X_test, y_test, config, fb.attacks.LinfDeepFoolAttack)
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
