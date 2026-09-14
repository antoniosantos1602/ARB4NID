from dataclasses import dataclass
import foolbox as fb
from models.attacks.base_attack import BaseAttack
from models.attacks.foolbox.attacks_foolbox import run_fb_logic

@dataclass
class PGDConfigFB:
    eps: float
    steps: int
    rel_stepsize: float
    targeted: bool = False

class PGDFoolbox(BaseAttack):
    def __init__(self):
        super().__init__(name="PGD", tool="FOOLBOX")

    def get_params(self, form) -> PGDConfigFB:
        try:
            return PGDConfigFB(
                eps=float(form.get("epsilon_pgd") or 0.2),
                steps=int(form.get("steps") or 50),
                rel_stepsize=float(form.get("rel_stepsize") or 0.1),
                targeted=False,
            )
        except ValueError:
            raise ValueError("Invalid PGD Foolbox parameters")

    def generate_x_adv(self, model, X_test, y_test, config: PGDConfigFB, **kwargs):
        try:
            return run_fb_logic(model, X_test, y_test, config, fb.attacks.LinfProjectedGradientDescentAttack)
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
