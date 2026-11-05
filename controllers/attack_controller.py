import logging

from models.data.feature_utils import *
from models.model_service import get_model_framework

import json
from flask import Response, request
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from utils.parallel_utils import run_single_attack, run_single_a2pm_attack, run_single_foolbox_attack
import pandas as pd
import numpy as np
from models.data.datasetInteligence import DatasetIntelligence
from models.container import get_intelligence_cache, get_results_repository
import os
import math
import multiprocessing

logger = logging.getLogger(__name__)

MAX_WORKERS = int(os.environ.get("ARB4NID_MAX_WORKERS", "6"))


def _json_safe(obj):
    if isinstance(obj, float) or isinstance(obj, np.floating):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj

GRADIENT_ATTACKS = {"FGM", "PGD", "BIM", "DeepFool", "JSMA", "C&W_L2"}
DIFFERENTIABLE_FRAMEWORKS = {"pytorch", "keras"}
TARGETED_PREFIX = {
    "FGM": "fgm", "PGD": "pgd", "BIM": "bim", "JSMA": "jsma",
    "C&W_L2": "cw", "ZOO": "zoo", "BoundaryAttack": "boundary", "HopSkipJump": "hsj",
}

def is_compatible(atk_name, atk_type, framework):
    if atk_type == "FOOLBOX":
        return framework in DIFFERENTIABLE_FRAMEWORKS
    if atk_type == "ART" and atk_name in GRADIENT_ATTACKS:
        return framework in DIFFERENTIABLE_FRAMEWORKS
    return True


class AttackController:

  @staticmethod
  def build_attack_result_json(result_dict):
        clean_detection = result_dict.get("clean_detection", {"tp": 0, "total": 0, "rate": 0.0})

        try:
            clean_acc = round(float(result_dict.get("clean_accuracy", 0.0)), 6)
            exec_time = round(float(result_dict.get("execution_time", 0.0)), 6)
        except (TypeError, ValueError):
            clean_acc = exec_time = 0.0

        return _json_safe({
            "model_name": result_dict.get("model_name"),
            "attack_name": result_dict.get("attack_name"),
            "params": result_dict.get("attack_params", {}),
            "clean_accuracy": clean_acc,
            "clean_detection": clean_detection,
            "execution_time": exec_time,
            "adv_acc_by_k_cumulativo": result_dict.get("adv_acc_by_k_cumulativo", {}),
            "adv_confusion_exact": result_dict.get("adv_confusion_exact", {}),
            "adv_confusion_global": result_dict.get("adv_confusion_global"),
            "class_labels": result_dict.get("class_labels", []),
            "binary_violations": result_dict.get("binary_violations", []),
            "integer_violations": result_dict.get("integer_violations", []),
            "mahalanobis_distribution": result_dict.get("mahalanobis_distribution", {}),
            "constraints_for_k": result_dict.get("constraints_for_k", {}),
            "targeted": result_dict.get("targeted", False),
            "target_class":result_dict.get("target_class"),
            "asr": result_dict.get("asr"),
            "adversarial_accuracy": result_dict.get("adversarial_accuracy"),
            "realistic_adversarial_accuracy": result_dict.get("realistic_adversarial_accuracy"),

            "wasr_v2":      result_dict.get("wasr_v2"),
            "n0":           result_dict.get("n0"),
            "n_k0_success": result_dict.get("n_k0_success"),
            "n_success":    result_dict.get("n_success"),
            "nk":           result_dict.get("nk", {}),
            "n_uni_total":  result_dict.get("n_uni_total"),
            "n_excl_total": result_dict.get("n_excl_total"),
            "n_lin_total":  result_dict.get("n_lin_total"),
            "dep_ausente":  result_dict.get("dep_ausente"),
            "penalty_percentiles": result_dict.get("penalty_percentiles", {}),
            "constraint_breach_rate": result_dict.get("constraint_breach_rate", {}),
            "nk_detail": result_dict.get("nk_detail", {}),
            "mahalanobis_zones": result_dict.get("mahalanobis_zones", {}),
            "n_binary_cols": result_dict.get("n_binary_cols", 0),
            "n_integer_cols": result_dict.get("n_integer_cols", 0),
        })

  @staticmethod
  def build_summary_json(result_dict):
        def fmt(val, suffix=""):
            try:
                return f"{round(float(val), 5)}{suffix}"
            except (TypeError, ValueError):
                return "N/A"

        return {
            "attack": result_dict.get("attack_name"),
            "model": result_dict.get("model_name"),
            "clean_accuracy":               fmt(result_dict.get("clean_accuracy"), "%"),
            "execution_time":               fmt(result_dict.get("execution_time"), "s"),
            "asr":                          fmt(result_dict.get("asr"), "%"),
            "adversarial_accuracy":         fmt(result_dict.get("adversarial_accuracy"), "%"),
            "attack_reputation_score":      fmt(result_dict.get("wasr_v2"), "%"),
            "realistic_adversarial_accuracy": fmt(result_dict.get("realistic_adversarial_accuracy"), "%"),
        }

  @staticmethod
  def get_summary_export():
        results = get_results_repository().all()
        return [AttackController.build_summary_json(res) for res in results]

  @staticmethod
  def get_all_results():
    results = get_results_repository().all()
    payload = []

    for res in results:
        data = AttackController.build_attack_result_json(res)

        data["attack_id"] = res.get("attack_id")

        payload.append(data)

    return payload


  @staticmethod
  def run_attacks_stream(form_data, test_path, model_paths):
      cache = get_intelligence_cache()
      intelligence = cache.get()
      logger.debug("intelligence retrieved from cache")

      if not intelligence:
            df_temp = pd.read_csv(test_path)
            intelligence = DatasetIntelligence(df_temp)
            cache.save(intelligence)

      json_file = request.files.get("attack_config_file")
      json_params = {}
      selected_attacks = request.form.getlist("selected_attacks")
      selected_attacks_targeted = request.form.getlist("selected_attacks_targeted")

      if json_file and json_file.filename != '':
        try:
            data = json.load(json_file)
            logger.info("JSON Config loaded")

            if "attacks" in data:
                if not selected_attacks:
                    selected_attacks = [atk["type"] for atk in data["attacks"]]

                for atk in data["attacks"]:
                    prefix = atk["type"].replace("_ART", "").replace("_FOOLBOX", "").lower()
                    for param_name, param_value in atk.get("params", {}).items():
                        json_params[f"{prefix}_{param_name}"] = param_value

        except Exception as e:
            logger.error("Error parsing organized JSON: %s", e, exc_info=True)

      combined_params = {**request.form.to_dict(), **json_params}
      df_original = intelligence._df_orig
      column_names = intelligence.feature_names

      X_test = df_original.drop(columns=['label']).values.astype(np.float32)
      y_test = df_original['label'].values.astype(int)
      logger.debug("X_test/y_test arrays prepared")
      shared_params = {
            "form": combined_params,
            "X_test": X_test,
            "y_test": y_test,
            "intelligence": intelligence,
            "df_original": df_original,
            "column_names": column_names,
      }
      logger.debug("shared_params built")

      targeted_art = [atk.replace("_ART", "") for atk in selected_attacks_targeted if atk.endswith("_ART")]
      targeted_form = {**combined_params}
      # O modo direcionado tem os seus próprios parâmetros, submetidos com o
      # prefixo "tgt__" (ex.: tgt__fgm_eps). Aqui cada override substitui o
      # <key> correspondente APENAS no form direcionado, deixando o não
      # direcionado intacto — as duas configurações são independentes.
      for _k, _v in request.form.to_dict().items():
          if _k.startswith("tgt__"):
              targeted_form[_k[len("tgt__"):]] = _v
      for atk in targeted_art:
          prefix = TARGETED_PREFIX.get(atk)
          if prefix:
              targeted_form[f"{prefix}_targeted"] = "True"
      if "A2PM" in selected_attacks_targeted:
          targeted_form["a2pm_targeted"] = "True"
      shared_params_targeted = {**shared_params, "form": targeted_form}

      results_repository = get_results_repository()

      def generate():
          try:
              logger.debug("SSE stream starting")
              yield f"event: status\ndata: {json.dumps({'message': 'Starting attack execution'})}\n\n".encode()
              attacks_art_unt = [atk.replace("_ART", "") for atk in selected_attacks if atk.endswith("_ART")]
              attacks_art_tgt = [atk.replace("_ART", "") for atk in selected_attacks_targeted if atk.endswith("_ART")]
              attacks_foolbox = [atk.replace("_FOOLBOX", "") for atk in selected_attacks if atk.endswith("_FOOLBOX")]
              include_a2pm_unt = "A2PM" in [atk.upper() for atk in selected_attacks]
              include_a2pm_tgt = "A2PM" in [atk.upper() for atk in selected_attacks_targeted]

              tasks = []
              for atk, model in product(attacks_art_unt, model_paths):
                  tasks.append(("ART", atk, model, "untargeted"))
              for atk, model in product(attacks_art_tgt, model_paths):
                  tasks.append(("ART", atk, model, "targeted"))

              for atk in attacks_foolbox:
                  tasks.append(("FOOLBOX", atk, model_paths, "untargeted"))

              if include_a2pm_unt:
                  tasks.append(("A2PM", "A2PM", model_paths, "untargeted"))
              if include_a2pm_tgt:
                  tasks.append(("A2PM", "A2PM", model_paths, "targeted"))

              yield f"event: status\ndata: {json.dumps({'message': 'Scheduling attack tasks'})}\n\n".encode()

              _spawn_ctx = multiprocessing.get_context("spawn")
              with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=_spawn_ctx) as executor:
                  futures = {}
                  for t in tasks:
                      atk_type, atk_name, model_data, mode = t
                      mode_label = " (Targeted)" if mode == "targeted" else ""

                      if atk_type == "ART":
                          sp = shared_params_targeted if mode == "targeted" else shared_params
                          model_label = os.path.basename(model_data)
                          fw = get_model_framework(model_data)
                          if not is_compatible(atk_name, atk_type, fw):
                              yield f"event: skipped\ndata: {json.dumps({'attack': atk_name + mode_label, 'model': model_label, 'reason': f'Attack requires differentiable model ({fw} not supported)'})}\n\n".encode()
                              continue
                          yield f"event: running\ndata: {json.dumps({'attack': atk_name + mode_label, 'model': model_label})}\n\n".encode()
                          futures[executor.submit(run_single_attack, (atk_name, model_data, sp))] = t

                      elif atk_type == "FOOLBOX":
                          compatible_models = []
                          for mp in model_data:
                              fw = get_model_framework(mp)
                              model_label = os.path.basename(mp)
                              if not is_compatible(atk_name, atk_type, fw):
                                  yield f"event: skipped\ndata: {json.dumps({'attack': atk_name, 'model': model_label, 'reason': f'Foolbox requires differentiable model ({fw} not supported)'})}\n\n".encode()
                              else:
                                  yield f"event: running\ndata: {json.dumps({'attack': atk_name, 'model': model_label})}\n\n".encode()
                                  compatible_models.append(mp)
                          if compatible_models:
                              futures[executor.submit(run_single_foolbox_attack, (atk_name, compatible_models, shared_params))] = t

                      elif atk_type == "A2PM":
                          sp = shared_params_targeted if mode == "targeted" else shared_params
                          for mp in model_data:
                              yield f"event: running\ndata: {json.dumps({'attack': atk_name + mode_label, 'model': os.path.basename(mp)})}\n\n".encode()
                          futures[executor.submit(
                              run_single_a2pm_attack,
                              (model_data, sp)
                          )] = t


                  for future in as_completed(futures):
                      atk_type, atk_name, model_info, mode = futures[future]
                      try:
                          result_list = future.result()

                          for res in result_list:
                              if mode == "targeted":
                                  res["attack_name"] = f"{res.get('attack_name', atk_name)} (Targeted)"
                              attack_id = results_repository.add(res)
                              payload = AttackController.build_attack_result_json(res)
                              payload["framework"] = atk_type
                              payload["attack_id"] = attack_id
                              yield f"event: result\ndata: {json.dumps(payload)}\n\n".encode()


                      except Exception as e:
                          model_name = os.path.basename(model_info) if isinstance(model_info, str) else "Model"
                          notification = {
                                        "is_notification": True,
                                        "attack": atk_name,
                                        "model": model_name,
                                        "error": str(e),
                                        "type": "warning"
                                    }
                          yield f"event: error\ndata: {json.dumps(notification)}\n\n".encode()

              yield "event: done\ndata: END\n\n".encode()


          except Exception as e:
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n".encode()

      return Response(
            generate(),
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            },
            direct_passthrough=True
      )


  @staticmethod
  def clear_all():
    try:
        get_results_repository().clear()


        return {"status": "ok", "message": "Cards cleared from memory"}, 200
    except Exception as e:
        logger.error("Error clearing: %s", e, exc_info=True)
        return {"status": "error", "message": str(e)}, 500

  @staticmethod
  def delete_result(attack_id):
    deleted = get_results_repository().delete(attack_id)

    if deleted:
        return {"status": "ok", "deleted": attack_id}, 200

    return {"status": "error", "message": "attack_id not found"}, 404
