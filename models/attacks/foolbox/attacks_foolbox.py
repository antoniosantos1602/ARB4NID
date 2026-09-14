import torch
import numpy as np
import tensorflow as tf
import eagerpy as ep
import foolbox as fb

def create_foolbox_model(model, bounds):
    if isinstance(model, tf.keras.Model):
        return fb.TensorFlowModel(model, bounds=bounds)
    elif isinstance(model, torch.nn.Module):
        return fb.PyTorchModel(model, bounds=bounds)
    else:
        raise ValueError("This model is not supported by Foolbox.")

def prepare_inputs(X, y, model_type):
    if model_type == "keras":
        x_tensor = tf.convert_to_tensor(X.astype(np.float32))
        y_tensor = tf.convert_to_tensor(y.astype(np.int64))
    else:
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.int64)
    return ep.astensors(x_tensor, y_tensor)

def run_fb_logic(model, X, y, config, attack_class):
    min_val, max_val = float(np.min(X)), float(np.max(X))
    fmodel = create_foolbox_model(model, bounds=(min_val, max_val))
    model_type = "keras" if isinstance(model, tf.keras.Model) else "pytorch"

    y_input = np.full(len(y), config.target_class) if config.targeted else y
    x_ep, y_ep = prepare_inputs(X, y_input, model_type)

    attack = attack_class()
    eps = getattr(config, "eps", 0.3)

    _, advs, _ = attack(fmodel, x_ep, y_ep, epsilons=[eps])

    return advs[0].raw.numpy() if model_type == "keras" else advs[0].raw.cpu().numpy()
