from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import torch


class ModelAdapter(ABC):
    name: str = ""

    @abstractmethod
    def matches(self, model) -> bool:
        pass

    @abstractmethod
    def predict(self, model, X) -> np.ndarray:
        pass

    def to_art_classifier(self, model, X, y):
        raise NotImplementedError(
            f"ART classifier creation not supported for {self.name!r}"
        )


class PyTorchAdapter(ModelAdapter):
    name = "pytorch"

    def matches(self, model) -> bool:
        return isinstance(model, torch.nn.Module)

    def predict(self, model, X) -> np.ndarray:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()
        return y_pred.astype(int)

    def to_art_classifier(self, model, X, y):
        import torch.nn as nn
        import torch.optim as optim
        from art.estimators.classification import PyTorchClassifier

        return PyTorchClassifier(
            model=model,
            loss=nn.CrossEntropyLoss(),
            optimizer=optim.Adam(model.parameters(), lr=0.01),
            input_shape=(X.shape[1],),
            nb_classes=len(set(y)),
            clip_values=(float(X.min()), float(X.max())),
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )


class KerasAdapter(ModelAdapter):
    name = "keras"

    def matches(self, model) -> bool:
        return isinstance(model, tf.keras.Model)

    def predict(self, model, X) -> np.ndarray:
        predictions = model.predict(X)
        y_pred = np.argmax(predictions, axis=1)
        return y_pred.astype(int)

    def to_art_classifier(self, model, X, y):
        from art.estimators.classification import TensorFlowV2Classifier

        return TensorFlowV2Classifier(
            model=model,
            loss_object=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            input_shape=(X.shape[1],),
            nb_classes=len(set(y)),
            clip_values=(float(X.min()), float(X.max())),
        )


class LightGBMAdapter(ModelAdapter):
    name = "lightgbm"

    def matches(self, model) -> bool:
        import lightgbm as lgb
        return isinstance(model, lgb.LGBMClassifier)

    def predict(self, model, X) -> np.ndarray:
        return model.predict(X).astype(int)

    def to_art_classifier(self, model, X, y):
        from art.estimators.classification import LightGBMClassifier
        return LightGBMClassifier(model=model, clip_values=(0, 1))


class XGBoostAdapter(ModelAdapter):
    name = "xgboost"

    def matches(self, model) -> bool:
        import xgboost as xgb
        return isinstance(model, xgb.XGBClassifier)

    def predict(self, model, X) -> np.ndarray:
        return model.predict(X).astype(int)

    def to_art_classifier(self, model, X, y):
        from art.estimators.classification import XGBoostClassifier
        return XGBoostClassifier(
            model=model,
            clip_values=(float(X.min()), float(X.max())),
            nb_features=X.shape[1],
            nb_classes=len(set(y)),
        )


class SklearnAdapter(ModelAdapter):
    name = "sklearn"

    def matches(self, model) -> bool:
        import sklearn.base
        return isinstance(model, sklearn.base.BaseEstimator)

    def predict(self, model, X) -> np.ndarray:
        return model.predict(X).astype(int)

    def to_art_classifier(self, model, X, y):
        from art.estimators.classification import SklearnClassifier
        return SklearnClassifier(model=model, clip_values=(0, 1))


class A2PMAdapter(ModelAdapter):
    name = "a2pm"

    def matches(self, model) -> bool:
        from a2pm.wrappers import TorchWrapper, KerasWrapper, SklearnWrapper
        return isinstance(model, (TorchWrapper, KerasWrapper, SklearnWrapper))

    def predict(self, model, X) -> np.ndarray:
        return model.predict(X)


ADAPTERS = [
    A2PMAdapter(),
    PyTorchAdapter(),
    KerasAdapter(),
    LightGBMAdapter(),
    XGBoostAdapter(),
    SklearnAdapter(),
]


def get_adapter(model) -> ModelAdapter:
    for adapter in ADAPTERS:
        if adapter.matches(model):
            return adapter
    raise ValueError(f"No adapter for model type {type(model).__name__}")


def get_framework_name(model) -> str:
    try:
        return get_adapter(model).name
    except ValueError:
        return "unknown"
