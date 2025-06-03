from art.estimators.classification import (
    SklearnClassifier, XGBoostClassifier, PyTorchClassifier,
    TensorFlowV2Classifier, LightGBMClassifier
)
import torch
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
import sklearn

import torch.nn as nn
import torch.optim as optim


def create_art_classifier(model, X, y, attack_type=None):

    nb_classes = len(set(y))
    input_shape = (X.shape[1],)

    white_box_attacks = {"FGM", "PGD", "DeepFool", "C&W_L2", "JSMA"}

    if isinstance(model, torch.nn.Module):
        return PyTorchClassifier(
            model=model,
            loss=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(model.parameters(), lr=0.01),
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_values=(0, 1),
            device_type="cuda" if torch.cuda.is_available() else "cpu"
        )

    elif isinstance(model, tf.keras.Model):
        return TensorFlowV2Classifier(
            model=model,
            loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_values=(0, 1)
        )

    elif isinstance(model, lgb.LGBMClassifier):
        print("1")
        if attack_type in white_box_attacks:
            raise ValueError(f"O ataque {attack_type} requer gradientes e não é compatível com LightGBM.")
        return LightGBMClassifier(model=model, clip_values=(0, 1))

    elif isinstance(model, xgb.XGBClassifier):
        if attack_type in white_box_attacks:
            raise ValueError(f"O ataque {attack_type} requer gradientes e não é compatível com XGBoost.")
        return XGBoostClassifier(
            model=model,
            clip_values=(0, 1),
            nb_features=input_shape[0],
            nb_classes=nb_classes
        )

    elif isinstance(model, sklearn.base.BaseEstimator):
        if attack_type in white_box_attacks:
            raise ValueError(f"O ataque {attack_type} requer gradientes e não é compatível com modelos do scikit-learn.")
        return SklearnClassifier(model=model, clip_values=(0, 1))

    else:
        raise ValueError("Tipo de modelo não suportado para ART.")
