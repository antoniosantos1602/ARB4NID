from models.attacks.art.jsma import JSMAAttack
from models.attacks.art.fgm import FGMAttack
from models.attacks.art.zoo import ZOOAttack
from models.attacks.art.pgd import PGDAttack
from models.attacks.art.boundary import BoundaryAttackClass
from models.attacks.art.deepfool import DeepFoolAttack
from models.attacks.art.hopskipjump import HopSkipJumpAttack
from models.attacks.art.carlini_and_wagner import CWAttackL2
from models.attacks.art.bim import BIMAttack
from models.attacks.a2pm.a2pm_attack import A2PMAttack

ATTACK_REGISTRY = {
    "JSMA": JSMAAttack(),
    "FGM": FGMAttack(),
    "ZOO": ZOOAttack(),
    "PGD": PGDAttack(),
    "BoundaryAttack": BoundaryAttackClass(),
    "DeepFool": DeepFoolAttack(),
    "HopSkipJump": HopSkipJumpAttack(),
    "C&W_L2": CWAttackL2(),
    "BIM": BIMAttack(),
    "A2PM": A2PMAttack()
}

def run_attack(attack_type,
               request,
               model_paths,
               X_test,
               y_test,
               df_original,
               column_names,
               intelligence):

    clean_name = attack_type.replace("_ART", "").replace("_FOOLBOX", "")
    if clean_name == "ZOO_TARGETED":
        clean_name = "ZOO"

    attack_instance = ATTACK_REGISTRY.get(clean_name)

    if not attack_instance:
        raise ValueError(f"Unsupported attack: {clean_name}")

    shared_params = {
        "X_test": X_test,
        "y_test": y_test,
        "df_original": df_original,
        "column_names": column_names,
        "intelligence": intelligence
    }

    return attack_instance.run(request.form, model_paths, shared_params)
