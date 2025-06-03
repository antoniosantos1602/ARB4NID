import numpy as np
import pandas as pd
import joblib
import torch
import tensorflow as tf

def load_dataset(train_path, test_path):
    target_column = 'label'

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Garantir que as labels estão corretamente mapeadas (caso venham como string)
    if train_df[target_column].dtype == object:
        train_df[target_column] = train_df[target_column].str.strip().str.lower()
        test_df[target_column] = test_df[target_column].str.strip().str.lower()

        label_map = {'benign': 0, 'malicious': 1}
        train_df[target_column] = train_df[target_column].map(label_map)
        test_df[target_column] = test_df[target_column].map(label_map)

    # Remover valores ausentes
    train_df = train_df.dropna(subset=[target_column])
    test_df = test_df.dropna(subset=[target_column])

    # Separar X e y
    X_train = train_df.drop(columns=[target_column]).astype(np.float32)
    y_train = train_df[target_column].astype(int)

    X_test = test_df.drop(columns=[target_column]).astype(np.float32)
    y_test = test_df[target_column].astype(int)

    return X_train, y_train, X_test, y_test

def load_model(model_path):
    if model_path.endswith('.joblib') or model_path.endswith('.pkl'):
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
    else:
        raise ValueError("Formato de modelo não suportado. Use apenas .joblib ou .pkl.")


def predict_model(model, X):

    if isinstance(model, tf.keras.Model):
        predictions = model.predict(X)
        y_pred = np.argmax(predictions, axis=1)  # Converter probabilidades para rótulos (0 ou 1)
        return y_pred.astype(int)  # Garantir que a saída é inteira

    elif isinstance(model, torch.nn.Module):  # Para modelos PyTorch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        if isinstance(X, np.ndarray):
            X = X.astype(np.float32)


        #print(f"X convertido para tensor PyTorch: {X.shape}, dtype: {X.dtype}")

        with torch.no_grad():
            outputs = model(X_tensor)
            y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

        return y_pred.astype(int)  # Garantir que a saída é inteira

    else:
        return model.predict(X).astype(int)  # Garantir que a saída é inteira