import numpy as np
import pandas as pd


class CsvExporter:
    def __init__(self, attack_result, X_orig, X_adv, features_names,
                 y_true, y_pred_clean, y_pred_adv, num_violations) -> None:
        self.res = attack_result
        self.X_orig = X_orig
        self.X_adv = X_adv
        self.features_names = features_names
        self.y_true = y_true
        self.y_pred_clean = y_pred_clean
        self.y_pred_adv = y_pred_adv
        self.num_violations = num_violations

    def export_all_success(self, mask, output_path) -> None:
        indices = np.where(mask)[0]

        if len(indices) == 0:
            pd.DataFrame({"info": ["No successful adversarial samples found"]}).to_csv(output_path, index=False)
            return

        validator = self.res.get("validator")
        has_deps = validator is not None and (
            len(validator.dependencias_unidirecionais) > 0
            or len(validator.exclusividades_mutuas) > 0
            or len(validator.dependencias_lineares) > 0
        )

        if has_deps:
            n_uni = validator._count_uni_violations(indices)
            n_excl = validator._count_excl_violations(indices)
            n_lin = validator._count_lin_violations(indices)
        else:
            n_uni = n_excl = n_lin = np.zeros(len(indices), dtype=int)

        if validator is not None:
            pen_bin = validator._get_binary_penalty_per_sample(indices)
            pen_int = validator._get_integer_penalty_per_sample(indices)
            validator._get_mask_mahalanobis_outliers(indices)
            pen_mah = validator._get_mahalanobis_penalty_per_sample(indices)
            pen_dep, dep_logs = validator._get_dependency_penalty_v2(indices)
            pen_uni = dep_logs.get("pen_uni", np.zeros(len(indices), dtype=float))
            pen_excl = dep_logs.get("pen_excl", np.zeros(len(indices), dtype=float))
            pen_lin = dep_logs.get("pen_lin", np.zeros(len(indices), dtype=float))
            dep_ausente = not has_deps
            pen_total = (
                (pen_bin + pen_int + pen_mah) / 3.0
                if dep_ausente
                else 0.25 * pen_bin + 0.25 * pen_int + 0.25 * pen_dep + 0.25 * pen_mah
            )
        else:
            n = len(indices)
            pen_bin = pen_int = pen_dep = pen_mah = pen_total = np.zeros(n, dtype=float)
            pen_uni = pen_excl = pen_lin = np.zeros(n, dtype=float)

        X_orig_values = self.X_orig[self.features_names].values
        rows = []
        for pos, i in enumerate(indices):
            row = {}
            for j, feat in enumerate(self.features_names):
                row[f"{feat}_orig"] = X_orig_values[i][j]
                row[f"{feat}_adv"] = self.X_adv[i][j]
            row.update({
                "y_true":          int(self.y_true[i]),
                "y_pred_clean":    int(self.y_pred_clean[i]),
                "y_pred_adv":      int(self.y_pred_adv[i]),
                "violations":      int(self.num_violations[i]),
                "n_uni_violadas":  int(n_uni[pos]),
                "n_excl_violadas": int(n_excl[pos]),
                "n_lin_violadas":  int(n_lin[pos]),
                "pen_bin":         round(float(pen_bin[pos]),  4),
                "pen_int":         round(float(pen_int[pos]),  4),
                "pen_dep_uni":     round(float(pen_uni[pos]),  4),
                "pen_dep_excl":    round(float(pen_excl[pos]), 4),
                "pen_dep_lin":     round(float(pen_lin[pos]),  4),
                "pen_dep":         round(float(pen_dep[pos]),  4),
                "pen_mah":         round(float(pen_mah[pos]),  4),
                "pen_total":       round(float(pen_total[pos]), 4),
            })
            rows.append(row)

        pd.DataFrame(rows).to_csv(output_path, index=False)
