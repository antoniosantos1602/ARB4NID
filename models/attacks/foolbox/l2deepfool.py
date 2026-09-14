from dataclasses import dataclass
import foolbox as fb
from models.attacks.base_attack import BaseAttack
from models.attacks.foolbox.attacks_foolbox import run_fb_logic

@dataclass
class DeepFoolConfigFB:
    steps: int
    overshoot: float
    targeted: bool = False

class DeepFoolFoolbox(BaseAttack):
    def __init__(self):
        super().__init__(name="L2DeepFool", tool="FOOLBOX")

    def get_params(self, form) -> DeepFoolConfigFB:
        try:
            return DeepFoolConfigFB(
                steps=int(form.get("steps") or 50),
                overshoot=float(form.get("overshoot") or 0.02),
                targeted=False,
            )
        except ValueError:
            raise ValueError("Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: DeepFoolConfigFB, **kwargs):
        try:
            return run_fb_logic(model, X_test, y_test, config, fb.attacks.L2DeepFoolAttack)
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
