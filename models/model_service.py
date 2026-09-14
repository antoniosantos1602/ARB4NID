import joblib

from models.model_adapters import get_adapter, get_framework_name


def get_model_framework(model_path):
    model, _ = load_model(model_path)
    return get_framework_name(model)


def load_model(model_path):
    if not model_path.endswith('.joblib'):
        raise ValueError("This template format is not supported. Please use only .joblib or .pkl.")

    loaded = joblib.load(model_path)

    if isinstance(loaded, tuple) and len(loaded) == 2:
        model, scaler = loaded
    elif isinstance(loaded, dict) and "model" in loaded:
        model = loaded["model"]
        scaler = loaded.get("scaler", None)
    else:
        model = loaded
        scaler = None

    return model, scaler


def predict_model(model, X):
    return get_adapter(model).predict(model, X)
