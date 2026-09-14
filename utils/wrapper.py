from models.model_adapters import get_adapter


def create_art_classifier(model, X, y, attack_type=None):
    return get_adapter(model).to_art_classifier(model, X, y)
