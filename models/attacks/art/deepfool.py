from dataclasses import dataclass
from art.attacks.evasion import DeepFool as ARTDeepFool
from utils.wrapper import create_art_classifier
from models.attacks.base_attack import BaseAttack

@dataclass
class DeepFoolConfig:
    max_iter: int
    epsilon: float
    batch_size: int
    targeted: bool = False

class DeepFoolAttack(BaseAttack):
    def __init__(self):
        super().__init__(name="DeepFool", tool="ART")

    def get_params(self, form) -> DeepFoolConfig:
        try:
            return DeepFoolConfig(
                max_iter=int(form.get("deepfool_max_iter") or 20),
                epsilon=float(form.get("deepfool_eps") or 1e-6),
                batch_size=int(form.get("deepfool_batch_size") or 32)
            )
        except ValueError:
            raise ValueError("Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: DeepFoolConfig, **kwargs):
        try:
            art_classifier = create_art_classifier(model, X_test, y_test, attack_type="DeepFool")

            attack = ARTDeepFool(
                classifier=art_classifier,
                max_iter=config.max_iter,
                epsilon=config.epsilon,
                nb_grads=min(10, art_classifier.nb_classes),
                batch_size=config.batch_size,
                verbose=True
            )

            X_adv = attack.generate(x=X_test)

            return X_adv
        except Exception as e:
            raise TypeError(f"This model is not supported for this type of attack.")
