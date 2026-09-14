import numpy as np
from dataclasses import dataclass
from art.attacks.evasion import CarliniL2Method
from art.utils import to_categorical
from utils.wrapper import create_art_classifier
from models.attacks.base_attack import BaseAttack

@dataclass
class CWConfig:
    confidence: float
    max_iter: int
    batch_size: int
    targeted: bool
    target_class: int

class CWAttackL2(BaseAttack):
    def __init__(self):
        super().__init__(name="C&W_L2", tool="ART")

    def get_params(self, form) -> CWConfig:
        try:
            raw_targeted = form.get("cw_targeted", "").strip().lower()
            targeted = raw_targeted in ("true", "1", "on", "yes")

            return CWConfig(
                confidence=float(form.get("cw_confidence") or 0.0),
                max_iter=int(form.get("cw_max_iter") or 10),
                batch_size=int(form.get("cw_batch_size") or 1),
                targeted=targeted,
                target_class=int(form.get("cw_target_class") or 0)
            )
        except ValueError:
            raise ValueError("Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: CWConfig, **kwargs):
        try:
            art_classifier = create_art_classifier(model, X_test, y_test, attack_type="C&W_L2")

            attack = CarliniL2Method(
                classifier=art_classifier,
                confidence=config.confidence,
                max_iter=config.max_iter,
                batch_size=config.batch_size,
                targeted=config.targeted,
                verbose=True
            )

            if config.targeted:
                n_classes = art_classifier.nb_classes
                y_target_oh = to_categorical(np.full(len(y_test), config.target_class), nb_classes=n_classes)
                X_adv = attack.generate(x=X_test, y=y_target_oh)
            else:
                X_adv = attack.generate(x=X_test)

            return X_adv
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
