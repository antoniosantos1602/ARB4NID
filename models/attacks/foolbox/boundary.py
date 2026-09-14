from dataclasses import dataclass
import foolbox as fb
from models.attacks.base_attack import BaseAttack
from models.attacks.foolbox.attacks_foolbox import run_fb_logic

@dataclass
class BoundaryConfigFB:
    steps: int
    targeted: bool = False

class BoundaryAttackFoolbox(BaseAttack):
    def __init__(self):
        super().__init__(name="BoundaryAttack", tool="FOOLBOX")

    def get_params(self, form) -> BoundaryConfigFB:
        try:
            return BoundaryConfigFB(
                steps=int(form.get("steps") or 25),
                targeted=False,
            )
        except ValueError:
            raise ValueError("Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: BoundaryConfigFB, **kwargs):
        try:
            return run_fb_logic(model, X_test, y_test, config, fb.attacks.BoundaryAttack)
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
