from dataclasses import dataclass
import foolbox as fb
from models.attacks.base_attack import BaseAttack
from models.attacks.foolbox.attacks_foolbox import run_fb_logic

@dataclass
class CWConfigFB:
    steps: int
    targeted: bool = False

class CarliniWagnerFoolbox(BaseAttack):
    def __init__(self):
        super().__init__(name="L2CarliniWagnerAttack", tool="FOOLBOX")

    def get_params(self, form) -> CWConfigFB:
        try:
            return CWConfigFB(
                steps=int(form.get("steps") or 100),
                targeted=False,
            )
        except ValueError:
            raise ValueError("Parâmetros do Carlini & Wagner Foolbox inválidos")

    def generate_x_adv(self, model, X_test, y_test, config: CWConfigFB, **kwargs):
        try:
            return run_fb_logic(model, X_test, y_test, config, fb.attacks.L2CarliniWagnerAttack)
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
