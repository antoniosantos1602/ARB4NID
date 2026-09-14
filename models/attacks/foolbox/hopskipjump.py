from dataclasses import dataclass
import foolbox as fb
from models.attacks.base_attack import BaseAttack
from models.attacks.foolbox.attacks_foolbox import run_fb_logic

@dataclass
class HopSkipJumpConfigFB:
    steps: int
    targeted: bool = False

class HopSkipJumpFoolbox(BaseAttack):
    def __init__(self):
        super().__init__(name="HopSkipJumpAttack", tool="FOOLBOX")

    def get_params(self, form) -> HopSkipJumpConfigFB:
        try:
            return HopSkipJumpConfigFB(
                steps=int(form.get("steps") or 40),
                targeted=False,
            )
        except ValueError:
            raise ValueError("Invalid HopSkipJump parameters")

    def generate_x_adv(self, model, X_test, y_test, config: HopSkipJumpConfigFB, **kwargs):
        try:
            return run_fb_logic(model, X_test, y_test, config, fb.attacks.HopSkipJumpAttack)
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
