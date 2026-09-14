from dataclasses import dataclass
import foolbox as fb
from models.attacks.base_attack import BaseAttack
from models.attacks.foolbox.attacks_foolbox import run_fb_logic

@dataclass
class BIMConfigFB:
    eps: float
    steps: int
    rel_stepsize: float
    targeted: bool = False

class BIMFoolbox(BaseAttack):
    def __init__(self):
        super().__init__(name="LinfBIM", tool="FOOLBOX")

    def get_params(self, form) -> BIMConfigFB:
        try:
            return BIMConfigFB(
                eps=float(form.get("epsilon") or 0.2),
                steps=int(form.get("steps") or 10),
                rel_stepsize=float(form.get("rel_stepsize") or 0.1),
                targeted=False,
            )
        except ValueError:
            raise ValueError("Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: BIMConfigFB, **kwargs):
        try:
            return run_fb_logic(model, X_test, y_test, config, fb.attacks.LinfBasicIterativeAttack)
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
