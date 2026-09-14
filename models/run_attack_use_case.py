import logging
import os
import time

import numpy as np
import pandas as pd

from models.constraints.adversarial_validation import AdversarialValidator
from models.metrics.metrics import (
    AdaptativeMetricV2,
    AdversarialContext,
    AdversarialMetrics,
    OriginalMetrics,
)
from models.model_service import load_model, predict_model

logger = logging.getLogger(__name__)


class RunAttackUseCase:

    def execute(
        self,
        attack_fn,
        attack_name,
        attack_tool,
        attack_params,
        model_paths,
        X_data,
        y_data,
        df_data_original,
        column_names,
        intelligence,
    ):
        return [
            self._run_for_model(
                attack_fn, attack_name, attack_tool, attack_params,
                model_path, X_data, y_data, df_data_original, column_names, intelligence,
            )
            for model_path in model_paths
        ]


    def _run_for_model(
        self, attack_fn, attack_name, attack_tool, attack_params,
        model_path, X_data, y_data, df_data_original, column_names, intelligence,
    ):
        model, scaler, model_name, X_scaled = self._load_and_scale(model_path, X_data)

        clean_accuracy, detection_rate, y_pred_clean = self._predict_clean(
            model, X_scaled, y_data
        )

        config_obj = attack_params.get("config")
        X_adv_denorm, X_adv_for_model, execution_time = self._run_attack(
            attack_fn, model, X_scaled, y_data, config_obj, column_names, scaler
        )

        y_pred_adv = predict_model(model, X_adv_for_model)

        is_targeted = config_obj.targeted
        target_class = getattr(config_obj, "target_class", None)

        context = AdversarialContext(
            y_true=y_data,
            y_pred_clean=y_pred_clean,
            targeted=is_targeted,
            target_class=target_class,
        )
        indices_do_universo = np.where(context.get_universe())[0]

        validator = AdversarialValidator(
            X_orig=df_data_original,
            X_adv=X_adv_denorm,
            success_indices=indices_do_universo,
            y_test=y_data,
            provider=intelligence,
            model=model,
            y_pred_adv=y_pred_adv,
        )
        logger.debug("validator instantiated")

        metrics = self._compute_metrics(
            context, validator, y_pred_adv, indices_do_universo, y_data
        )

        attack_config_vars = (
            {k: v for k, v in vars(config_obj).items() if not k.startswith("_")}
            if config_obj else {}
        )

        return self._build_result_dict(
            model_name=model_name,
            attack_name=attack_name,
            attack_tool=attack_tool,
            execution_time=execution_time,
            clean_accuracy=clean_accuracy,
            detection_rate=detection_rate,
            y_data=y_data,
            y_pred_clean=y_pred_clean,
            y_pred_adv=y_pred_adv,
            X_adv_denorm=X_adv_denorm,
            column_names=column_names,
            df_data_original=df_data_original,
            context=context,
            validator=validator,
            indices_do_universo=indices_do_universo,
            is_targeted=is_targeted,
            target_class=target_class,
            attack_config_vars=attack_config_vars,
            metrics=metrics,
        )

    def _load_and_scale(self, model_path, X_data):
        model, scaler = load_model(model_path)
        model_name = os.path.basename(model_path)
        X_scaled = (
            scaler.transform(pd.DataFrame(X_data, columns=scaler.feature_names_in_))
            if scaler else X_data
        )
        logger.debug("model loaded: %s", model_name)
        return model, scaler, model_name, X_scaled

    def _predict_clean(self, model, X_scaled, y_data):
        y_pred_clean = predict_model(model, X_scaled)
        orig = OriginalMetrics(y_true=y_data, y_pred_clean=y_pred_clean)
        logger.debug("clean evaluation done")
        return (
            orig.clean_accuracy(),
            {"rate": orig.detection_rate(benign_class_label=0)},
            y_pred_clean,
        )

    def _run_attack(self, attack_fn, model, X_scaled, y_data, config_obj, column_names, scaler):
        logger.debug("attack config: %s", config_obj)
        start = time.perf_counter()
        X_adv = attack_fn(
            model=model,
            X_test=X_scaled,
            y_test=y_data,
            config=config_obj,
            column_names=column_names,
        )
        execution_time = round(time.perf_counter() - start, 2)
        X_adv_denorm = scaler.inverse_transform(X_adv) if scaler else X_adv
        X_adv_for_model = (
            scaler.transform(pd.DataFrame(X_adv_denorm, columns=scaler.feature_names_in_))
            if scaler else X_adv_denorm
        )
        return X_adv_denorm, X_adv_for_model, execution_time

    def _compute_metrics(self, context, validator, y_pred_adv, indices_do_universo, y_data):
        num_violacoes_all = np.zeros(len(y_data), dtype=int)
        labels = [int(x) for x in sorted(np.unique(y_data))]

        adv_metrics = AdversarialMetrics(
            context=context,
            y_pred_adv=y_pred_adv,
            num_breaches=num_violacoes_all,
        )
        adv_asr = adv_metrics.attack_success_rate()
        adv_acc = adv_metrics.adversarial_accuracy()
        adv_confusion_global = adv_metrics.confusion_matrix_global(labels=labels)
        logger.debug("ASR=%.2f AA=%.2f", adv_asr, adv_acc)

        logger.debug("computing adaptive metric V2")
        wasr_v2_result = AdaptativeMetricV2().compute_wasr_v2(
            context=context,
            validator=validator,
            universe_indices=indices_do_universo,
            y_pred_adv=y_pred_adv,
        )

        num_violacoes_all[indices_do_universo] = (
            (wasr_v2_result["pen_bin"] > 0).astype(int)
            + (wasr_v2_result["pen_int"] > 0).astype(int)
            + (wasr_v2_result["pen_dep"] > 0).astype(int)
            + (wasr_v2_result["pen_mah"] > 0).astype(int)
        )
        adv_metrics.num_breaches = num_violacoes_all

        adv_acc_by_k = adv_metrics.adversarial_accuracy_cumulative(max_k=4)
        adv_acc_by_k_json = {
            str(k): {
                "acc": v["acc"],
                "total": v["total"],
                "corretos": v["correct"],
                "enganados": v["misclassified"],
                "desvios": v["deviations"],
                "desvios_classes": v["deviation_classes"],
            }
            for k, v in adv_acc_by_k.items()
        }

        logger.debug("pipeline complete")
        return {
            "asr": adv_asr,
            "adversarial_accuracy": adv_acc,
            "adv_confusion_global": adv_confusion_global,
            "labels": labels,
            "num_violacoes_all": num_violacoes_all,
            "adv_acc_by_k_json": adv_acc_by_k_json,
            "wasr_v2_result": wasr_v2_result,
        }

    def _build_result_dict(
        self, model_name, attack_name, attack_tool, execution_time,
        clean_accuracy, detection_rate, y_data, y_pred_clean, y_pred_adv,
        X_adv_denorm, column_names, df_data_original, context, validator,
        indices_do_universo, is_targeted, target_class, attack_config_vars, metrics,
    ):
        wv2 = metrics["wasr_v2_result"]
        return {
            "model_name":               model_name,
            "attack_name":              attack_name,
            "attack_tool":              attack_tool,
            "execution_time":           execution_time,
            "clean_accuracy":           clean_accuracy,
            "clean_detection":          detection_rate,
            "success_indices":          indices_do_universo.tolist(),
            "adv_acc_by_k_cumulativo":  metrics["adv_acc_by_k_json"],
            "validator":                validator,
            "df_data_original":         df_data_original,
            "num_violacoes_all":        metrics["num_violacoes_all"].tolist(),
            "adv_confusion_exact":      {},
            "adv_confusion_global":     metrics["adv_confusion_global"],
            "class_labels":             metrics["labels"],
            "X_adv":                    X_adv_denorm,
            "column_names":             column_names,
            "y_true":                   y_data,
            "y_pred_clean":             y_pred_clean,
            "y_pred_adv":               y_pred_adv,
            "mask_iniciais":            context.mask_correct_clean.tolist(),
            "targeted":                 is_targeted,
            "target_class":             target_class,
            "X_orig_values":            df_data_original[column_names].values,
            "constraints_for_k":        {},
            "attack_params":            attack_config_vars,
            "asr":                      metrics["asr"],
            "adversarial_accuracy":     metrics["adversarial_accuracy"],
            "realistic_adversarial_accuracy": round((1 - wv2["wasr_v2"]) * 100, 5),
            "wasr_v2":                  wv2["wasr_v2"],
            "n0":                       wv2["n0"],
            "n_k0_success":             wv2["n_k0_success"],
            "n_success":                wv2["n_success"],
            "nk":                       wv2["nk"],
            "n_uni_total":              wv2["n_uni_total"],
            "n_excl_total":             wv2["n_excl_total"],
            "n_lin_total":              wv2["n_lin_total"],
            "dep_ausente":              wv2["dep_ausente"],
            "penalty_percentiles":      wv2.get("penalty_percentiles", {}),
            "constraint_breach_rate":   wv2.get("constraint_breach_rate", {}),
            "nk_detail":                wv2.get("nk_detail", {}),
            "mahalanobis_zones":        wv2.get("mahalanobis_zones", {}),
            "n_binary_cols":            len(validator.binary_cols),
            "n_integer_cols":           len(validator.integer_cols),
        }
