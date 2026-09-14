from dataclasses import dataclass
import foolbox as fb
from models.attacks.base_attack import BaseAttack
from models.attacks.foolbox.attacks_foolbox import run_fb_logic

@dataclass
class FGSMConfigFB:
    eps: float
    targeted: bool = False

class FGSMFoolbox(BaseAttack):
    def __init__(self):
        super().__init__(name="FGSM", tool="FOOLBOX")

    def get_params(self, form) -> FGSMConfigFB:
        try:
            return FGSMConfigFB(
                eps=float(form.get("epsilon") or 0.2),
                targeted=False,
            )
        except ValueError:
            raise ValueError("Parâmetros do FGSM Foolbox inválidos")

    def generate_x_adv(self, model, X_test, y_test, config: FGSMConfigFB, **kwargs):
        try:
            return run_fb_logic(model, X_test, y_test, config, fb.attacks.LinfFastGradientAttack)
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
