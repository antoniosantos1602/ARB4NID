from models.run_attack_use_case import RunAttackUseCase


def executar_ataque_realista(
    attack_fn,
    attack_params,
    model_paths,
    X_data,
    y_data,
    df_data_original,
    column_names,
    intelligence,
    **kwargs,
):
    attack_name = attack_params.pop("attack_name", "UNKNOWN")
    attack_tool = attack_params.pop("attack_tool", "ART")
    return RunAttackUseCase().execute(
        attack_fn, attack_name, attack_tool, attack_params,
        model_paths, X_data, y_data, df_data_original, column_names, intelligence,
    )
