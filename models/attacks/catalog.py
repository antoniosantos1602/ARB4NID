ATTACKS_CATALOG = [
    {
        "id": "FGM_ART", "name": "FGM", "label": "FGM (Fast Gradient Method)", "framework": "ART",
        "requires_gradient": True,
        "params": [
            {"key": "fgm_eps",        "label": "Epsilon",     "type": "float", "default": 0.2},
            {"key": "fgm_eps_step",   "label": "Eps Step",    "type": "float", "default": 0.2},
            {"key": "fgm_batch_size", "label": "Batch Size",  "type": "int",   "default": 128},
            {"key": "fgm_norm",       "label": "Norm",        "type": "str",   "default": "inf"},
            {"key": "fgm_targeted",   "label": "Targeted",    "type": "bool",  "default": False},
            {"key": "fgm_target_class","label":"Target Class","type": "int",   "default": 0},
        ],
    },
    {
        "id": "PGD_ART", "name": "PGD", "label": "PGD (Projected Gradient Descent)", "framework": "ART",
        "requires_gradient": True,
        "params": [
            {"key": "pgd_eps",          "label": "Epsilon",       "type": "float", "default": 0.2},
            {"key": "pgd_eps_step",     "label": "Eps Step",      "type": "float", "default": 0.1},
            {"key": "pgd_max_iter",     "label": "Max Iter",      "type": "int",   "default": 50},
            {"key": "pgd_batch_size",   "label": "Batch Size",    "type": "int",   "default": 128},
            {"key": "num_random_init_pgd","label":"Random Inits", "type": "int",   "default": 10},
            {"key": "pgd_norm",         "label": "Norm",          "type": "str",   "default": "inf"},
            {"key": "pgd_targeted",     "label": "Targeted",      "type": "bool",  "default": False},
            {"key": "pgd_target_class", "label": "Target Class",  "type": "int",   "default": 0},
        ],
    },
    {
        "id": "JSMA_ART", "name": "JSMA", "label": "JSMA (Saliency Map Attack)", "framework": "ART",
        "requires_gradient": True,
        "params": [
            {"key": "jsma_theta",       "label": "Theta",       "type": "float", "default": 0.2},
            {"key": "jsma_gamma",       "label": "Gamma",       "type": "float", "default": 0.2},
            {"key": "jsma_batch_size",  "label": "Batch Size",  "type": "int",   "default": 128},
            {"key": "jsma_targeted",    "label": "Targeted",    "type": "bool",  "default": False},
            {"key": "jsma_target_class","label": "Target Class","type": "int",   "default": 0},
        ],
    },
    {
        "id": "DeepFool_ART", "name": "DeepFool", "label": "DeepFool", "framework": "ART",
        "requires_gradient": True,
        "params": [
            {"key": "deepfool_max_iter",  "label": "Max Iter",  "type": "int",   "default": 100},
            {"key": "deepfool_eps",       "label": "Overshoot", "type": "float", "default": 0.000001},
            {"key": "deepfool_batch_size","label": "Batch Size","type": "int",   "default": 128},
        ],
    },
    {
        "id": "C&W_L2_ART", "name": "C&W_L2", "label": "C&W L2 (Carlini & Wagner)", "framework": "ART",
        "requires_gradient": True,
        "params": [
            {"key": "cw_confidence",    "label": "Confidence",    "type": "float", "default": 0.0},
            {"key": "cw_lr",            "label": "Learning Rate", "type": "float", "default": 0.01},
            {"key": "cw_max_iter",      "label": "Max Iter",      "type": "int",   "default": 10},
            {"key": "cw_batch_size",    "label": "Batch Size",    "type": "int",   "default": 256},
            {"key": "cw_targeted",      "label": "Targeted",      "type": "bool",  "default": False},
            {"key": "cw_target_class",  "label": "Target Class",  "type": "int",   "default": 0},
        ],
    },
    {
        "id": "ZOO_ART", "name": "ZOO", "label": "Zeroth Order Optimization (ZOO)", "framework": "ART",
        "requires_gradient": False,
        "params": [
            {"key": "zoo_max_iter",            "label": "Max Iter",          "type": "int",   "default": 10},
            {"key": "zoo_binary_search_steps", "label": "Binary Search Steps","type": "int",  "default": 1},
            {"key": "zoo_confidence",          "label": "Confidence",        "type": "float", "default": 0.0},
            {"key": "zoo_learning_rate",       "label": "Learning Rate",     "type": "float", "default": 0.5},
            {"key": "zoo_targeted",            "label": "Targeted",          "type": "bool",  "default": False},
            {"key": "zoo_target_class",        "label": "Target Class",      "type": "int",   "default": 0},
        ],
    },
    {
        "id": "BoundaryAttack_ART", "name": "BoundaryAttack", "label": "Boundary Attack", "framework": "ART",
        "requires_gradient": False,
        "params": [
            {"key": "boundary_max_iter", "label": "Max Iter",   "type": "int",   "default": 50},
            {"key": "boundary_batch_size","label": "Batch Size","type": "int",   "default": 128},
            {"key": "boundary_targeted", "label": "Targeted",   "type": "bool",  "default": False},
            {"key": "boundary_target_class","label":"Target Class","type":"int",  "default": 0},
        ],
    },
    {
        "id": "HopSkipJump_ART", "name": "HopSkipJump", "label": "HopSkipJump", "framework": "ART",
        "requires_gradient": False,
        "params": [
            {"key": "adv_max_iter",  "label": "Max Iter",  "type": "int", "default": 10},
            {"key": "max_eval",      "label": "Max Eval",  "type": "int", "default": 10},
            {"key": "init_eval",     "label": "Init Eval", "type": "int", "default": 20},
            {"key": "init_size",     "label": "Init Size", "type": "int", "default": 10},
            {"key": "hsj_batch_size","label": "Batch Size","type": "int", "default": 256},
            {"key": "hsj_norm",      "label": "Norm",      "type": "str", "default": "inf"},
            {"key": "hsj_targeted",  "label": "Targeted",  "type": "bool","default": False},
            {"key": "hsj_target_class","label":"Target Class","type":"int","default": 0},
        ],
    },
    {
        "id": "FGM_FOOLBOX", "name": "FGM", "framework": "FOOLBOX",
        "requires_gradient": True,
        "params": [
            {"key": "foolbox_epsilon", "label": "Epsilon", "type": "float", "default": 0.03},
        ],
    },
    {
        "id": "FGSM_FOOLBOX", "name": "FGSM", "framework": "FOOLBOX",
        "requires_gradient": True,
        "params": [
            {"key": "foolbox_epsilon", "label": "Epsilon", "type": "float", "default": 0.03},
        ],
    },
    {
        "id": "PGD_FOOLBOX", "name": "PGD", "framework": "FOOLBOX",
        "requires_gradient": True,
        "params": [
            {"key": "foolbox_epsilon",      "label": "Epsilon",      "type": "float", "default": 0.03},
            {"key": "foolbox_steps",        "label": "Steps",        "type": "int",   "default": 40},
            {"key": "foolbox_rel_stepsize", "label": "Rel Stepsize", "type": "float", "default": 0.025},
        ],
    },
    {
        "id": "L2PGD_FOOLBOX", "name": "L2PGD", "framework": "FOOLBOX",
        "requires_gradient": True,
        "params": [
            {"key": "foolbox_epsilon",      "label": "Epsilon",      "type": "float", "default": 0.5},
            {"key": "foolbox_steps",        "label": "Steps",        "type": "int",   "default": 40},
            {"key": "foolbox_rel_stepsize", "label": "Rel Stepsize", "type": "float", "default": 0.025},
        ],
    },
    {
        "id": "LinfBIM_FOOLBOX", "name": "LinfBIM", "framework": "FOOLBOX",
        "requires_gradient": True,
        "params": [
            {"key": "foolbox_epsilon",      "label": "Epsilon",      "type": "float", "default": 0.03},
            {"key": "foolbox_steps",        "label": "Steps",        "type": "int",   "default": 10},
            {"key": "foolbox_rel_stepsize", "label": "Rel Stepsize", "type": "float", "default": 0.025},
        ],
    },
    {
        "id": "L2CarliniWagnerAttack_FOOLBOX", "name": "L2CarliniWagnerAttack", "framework": "FOOLBOX",
        "requires_gradient": True,
        "params": [
            {"key": "foolbox_steps", "label": "Steps", "type": "int", "default": 1000},
        ],
    },
    {
        "id": "HopSkipJumpAttack_FOOLBOX", "name": "HopSkipJumpAttack", "framework": "FOOLBOX",
        "requires_gradient": True,
        "params": [
            {"key": "foolbox_steps", "label": "Steps", "type": "int", "default": 64},
        ],
    },
    {
        "id": "L2DeepFool_FOOLBOX", "name": "L2DeepFool", "framework": "FOOLBOX",
        "requires_gradient": True,
        "params": [
            {"key": "foolbox_steps",     "label": "Steps",     "type": "int",   "default": 50},
            {"key": "foolbox_overshoot", "label": "Overshoot", "type": "float", "default": 0.02},
        ],
    },
    {
        "id": "LinfDeepFool_FOOLBOX", "name": "LinfDeepFool", "framework": "FOOLBOX",
        "requires_gradient": True,
        "params": [
            {"key": "foolbox_steps",     "label": "Steps",     "type": "int",   "default": 50},
            {"key": "foolbox_overshoot", "label": "Overshoot", "type": "float", "default": 0.02},
        ],
    },
    {
        "id": "BoundaryAttack_FOOLBOX", "name": "BoundaryAttack", "framework": "FOOLBOX",
        "requires_gradient": False,
        "params": [
            {"key": "foolbox_steps", "label": "Steps", "type": "int", "default": 25000},
        ],
    },
    {
        "id": "A2PM", "name": "A2PM", "framework": "A2PM",
        "requires_gradient": False,
        "params": [],
    },
]
