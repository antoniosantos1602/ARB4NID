from dataclasses import dataclass
import foolbox as fb
from models.attacks.base_attack import BaseAttack
from models.attacks.foolbox.attacks_foolbox import run_fb_logic

@dataclass
class L2PGDConfigFB:
    eps: float
    steps: int
    rel_stepsize: float
    targeted: bool = False

class L2PGDFoolbox(BaseAttack):
    def __init__(self):
        super().__init__(name="L2PGD", tool="FOOLBOX")

    def get_params(self, form) -> L2PGDConfigFB:
        try:
            return L2PGDConfigFB(
                eps=float(form.get("epsilon_pgd") or 0.2),
                steps=int(form.get("steps") or 40),
                rel_stepsize=float(form.get("rel_stepsize") or 0.1),
                targeted=False,
            )
        except ValueError:
            raise ValueError("Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: L2PGDConfigFB, **kwargs):
        try:
            return run_fb_logic(model, X_test, y_test, config, fb.attacks.L2ProjectedGradientDescentAttack)
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
