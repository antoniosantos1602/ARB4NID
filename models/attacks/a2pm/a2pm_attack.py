import numpy as np
import torch
import tensorflow as tf
from dataclasses import dataclass
from typing import List, Dict, Any

from a2pm import A2PMethod
from a2pm.patterns import IntervalPattern, CombinationPattern
from a2pm.wrappers import TorchWrapper, KerasWrapper, SklearnWrapper
from models.attacks.base_attack import BaseAttack
from models.attacks.art.parse_params import parse_features_to_indices

@dataclass
class A2PMConfig:
    targeted: bool
    target_class: int
    iterations: int
    patience: int
    patterns_data: List[Dict[str, Any]]

class A2PMAttack(BaseAttack):
    def __init__(self):
        super().__init__(name="A2PM", tool="A2PM")

    def get_params(self, form) -> A2PMConfig:
        def extract_num(val, dtype=float, default=0.0):
            if isinstance(val, list):
                val = val[0]
            try:
                return dtype(val) if val and str(val).strip() != '' else default
            except:
                return default

        raw_targeted = form.get("a2pm_targeted", "False").strip().lower()
        patterns_raw = []

        if "use_combination" in form:
            pattern_indices = set()
            for key in form.keys():
                if key.startswith("combination_features_"):
                    idx_str = key.replace("combination_features_", "")
                    if idx_str.isdigit():
                        pattern_indices.add(int(idx_str))

            pattern_indices = sorted(pattern_indices)
            print(f"\n{len(pattern_indices)} padrões: {pattern_indices}")
            for idx in pattern_indices:
                feat_str = form.get(f"combination_features_{idx}", "")
                lock_str = form.get(f"locked_features_{idx}", "")
                prob_str = form.get(f"combination_probability_{idx}", "0.5")

                feat_list = [f.strip() for f in feat_str.split(',') if f.strip()]
                lock_list = [l.strip() for l in lock_str.split(',') if l.strip()]

                if feat_list:
                    patterns_raw.append({
                        "type": "combination",
                        "features": feat_list,
                        "locked": lock_list,
                        "prob": extract_num(prob_str, float, 0.5)
                    })
                    print(f"{idx}: {len(feat_list)} features, {len(lock_list)} locked")

        if "use_interval" in form:
            feat_str = form.get("features_int", "")
            feat_list = [f.strip() for f in feat_str.split(',') if f.strip()]

            if feat_list:
                int_str = form.get("integer_features", "")
                int_list = [i.strip() for i in int_str.split(',') if i.strip()]

                patterns_raw.append({
                    "type": "interval",
                    "features": feat_list,
                    "integer": int_list,
                    "ratio": extract_num(form.get("ratio"), float, 0.05),
                    "max_ratio": extract_num(form.get("max_ratio"), float, 0.1),
                    "missing": extract_num(form.get("missing_value"), float, 0.0),
                    "prob": extract_num(form.get("interval_probability"), float, 0.4)
                })
                print(f"{len(feat_list)} features")

        return A2PMConfig(
            targeted=raw_targeted in ("true", "1", "on", "yes"),
            target_class=extract_num(form.get("a2pm_target_class"), int, 0),
            iterations=extract_num(form.get("a2pm_iterations"), int, 5),
            patience=extract_num(form.get("a2pm_patience"), int, 2),
            patterns_data=patterns_raw
        )


    def generate_x_adv(self, model, X_test, y_test, config: A2PMConfig, column_names=None):
        final_patterns = []

        for p in config.patterns_data:
            try:
                f_idx = parse_features_to_indices(p["features"], column_names)
                if not f_idx:
                    continue

                if p["type"] == "combination":
                    l_idx = parse_features_to_indices(p["locked"], column_names) if p["locked"] else []
                    final_patterns.append(CombinationPattern(
                        features=[int(i) for i in f_idx],
                        locked_features=[int(i) for i in l_idx] if l_idx else None,
                        probability=float(p["prob"])
                    ))

                elif p["type"] == "interval":
                    i_idx = parse_features_to_indices(p["integer"], column_names) if p.get("integer") else []
                    final_patterns.append(IntervalPattern(
                        features=[int(i) for i in f_idx],
                        integer_features=[int(i) for i in i_idx] if i_idx else None,
                        ratio=p["ratio"],
                        max_ratio=p["max_ratio"],
                        missing_value=p["missing"],
                        probability=p["prob"]
                    ))
            except Exception as e:
                print(f"[A2PM] Error processing pattern {p['type']}: {e}")

        if not final_patterns:
            return X_test

        if isinstance(model, torch.nn.Module):
            model_w = TorchWrapper(model)
        elif isinstance(model, tf.keras.Model):
            model_w = KerasWrapper(model)
        else:
            model_w = SklearnWrapper(model)

        attack = A2PMethod(tuple(final_patterns))

        if config.targeted:
            y_np = np.array(y_test)
            mask = (y_np != config.target_class)
            if not np.any(mask): return X_test

            X_adv_p = attack.fit_generate(
                classifier=model_w, X=X_test[mask], y=y_np[mask],
                y_target=np.full_like(y_np[mask], config.target_class),
                iterations=config.iterations, patience=config.patience
            )
            X_adv = X_test.copy()
            X_adv[mask] = X_adv_p
            return X_adv

        return attack.fit_generate(
            classifier=model_w, X=X_test, y=y_test,
            iterations=config.iterations, patience=config.patience
        )
