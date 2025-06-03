import os
from flask import request
from sklearn.metrics import accuracy_score
from app.utils.utils import load_dataset, load_model, predict_model

def handle_upload():
    # Guardar os ficheiros
    train = request.files['train']
    test = request.files['test']
    model_file = request.files['model']

    os.makedirs('uploads', exist_ok=True)
    train_path = os.path.join('uploads', 'train.csv')
    test_path = os.path.join('uploads', 'test.csv')
    model_path = os.path.join('uploads', 'model.joblib')

    train.save(train_path)
    test.save(test_path)
    model_file.save(model_path)

    # Carregar dados e modelo
    X_train, y_train, X_test, y_test = load_dataset(train_path, test_path)
    model, scaler = load_model(model_path)

    if scaler:
        X_test = scaler.transform(X_test)

    # Fazer predição com função genérica
    y_pred = predict_model(model, X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Tamanho do treino: {X_train.shape[0]}")
    print(f"Tamanho do teste: {X_test.shape[0]}")
    print(f"Accuracy limpa no teste: {acc:.4f}")

    return "Ficheiros carregados e avaliados com sucesso!", 200
