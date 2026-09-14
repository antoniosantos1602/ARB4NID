from models.attacks.foolbox.fgm import FGMFoolbox
from models.attacks.foolbox.fgsm import FGSMFoolbox
from models.attacks.foolbox.boundary import BoundaryAttackFoolbox
from models.attacks.foolbox.bim import BIMFoolbox
from models.attacks.foolbox.carlini import CarliniWagnerFoolbox
from models.attacks.foolbox.hopskipjump import HopSkipJumpFoolbox
from models.attacks.foolbox.l2pgd import L2PGDFoolbox
from models.attacks.foolbox.pgd import PGDFoolbox
from models.attacks.foolbox.l2deepfool import DeepFoolFoolbox
from models.attacks.foolbox.linfdeepfool import LinfDeepFoolFoolbox

FOOLBOX_ATTACKS = {
    "FGM": FGMFoolbox(),
    "FGSM": FGSMFoolbox(),
    "BoundaryAttack": BoundaryAttackFoolbox(),
    "LinfBIM": BIMFoolbox(),
    "L2CarliniWagnerAttack": CarliniWagnerFoolbox(),
    "HopSkipJumpAttack": HopSkipJumpFoolbox(),
    "L2PGD": L2PGDFoolbox(),
    "PGD": PGDFoolbox(),
    "L2DeepFool": DeepFoolFoolbox(),
    "LinfDeepFool": LinfDeepFoolFoolbox(),
}

def run_foolbox_attack_router(
    attack_types,
    request,
    model_paths,
    X_test,
    y_test,
    df_original,
    column_names,
    intelligence,
):

    results = []

    shared_params = {
        "X_test": X_test,
        "y_test": y_test,
        "df_original": df_original,
        "column_names": column_names,
        "intelligence": intelligence
    }

    for attack_name in attack_types:
        if attack_name in FOOLBOX_ATTACKS:
            attack_obj = FOOLBOX_ATTACKS[attack_name]

            res = attack_obj.run(request.form, model_paths, shared_params)

            if res:
                results.extend(res)
        else:
            import logging
            logging.getLogger(__name__).warning("Unmapped Foolbox attack: %s", attack_name)

    return results
