# app/utils/param_utils.py

def get_param(params, key, default, expected_type=float, min_val=None, max_val=None):
    """
    Extrai e valida um parâmetro do dicionário `params`.

    :param params: dicionário recebido do frontend
    :param key: nome do parâmetro (ex: "epsilon")
    :param default: valor por defeito caso não exista
    :param expected_type: tipo esperado (float, int, etc)
    :param min_val: valor mínimo permitido (ou None)
    :param max_val: valor máximo permitido (ou None)
    :return: valor validado ou default
    """
    try:
        value = expected_type(params.get(key, default))

        if min_val is not None and value < min_val:
            raise ValueError(f"{key} abaixo do mínimo ({min_val})")
        if max_val is not None and value > max_val:
            raise ValueError(f"{key} acima do máximo ({max_val})")

        return value

    except (TypeError, ValueError):
        print(f"[WARN] Parâmetro inválido para '{key}', a usar default: {default}")
        return default
