import numpy as np
from sklearn.metrics import confusion_matrix

class OriginalMetrics:
    def __init__(self, y_true, y_pred_clean):
        self.y_true = np.array(y_true)
        self.y_pred_clean = np.array(y_pred_clean)
        self.total_samples = len(y_true)

    def clean_accuracy(self):
        if self.total_samples == 0:
            return 0.0
        correct_predictions = np.sum(self.y_true == self.y_pred_clean)
        return round(100 * (correct_predictions / self.total_samples), 5)

    def detection_rate(self, benign_class_label=0):
        is_attack = self.y_true != benign_class_label
        total_attacks = np.sum(is_attack)

        if total_attacks == 0:
            return 0.0

        correctly_classified_attacks = np.sum(is_attack & (self.y_pred_clean == self.y_true))
        return round(100 * (correctly_classified_attacks / total_attacks), 5)

class AdversarialContext:
    def __init__(self,y_true, y_pred_clean, targeted=False, target_class = None):
        self.y_true = np.asarray(y_true)
        self.y_pred_clean = np.asarray(y_pred_clean)
        self.targeted = targeted
        self.target_class = target_class
        self.mask_correct_clean = (self.y_pred_clean == self.y_true)

    def get_universe(self):
        if self.targeted:
            return self.mask_correct_clean & (self.y_true != self.target_class)
        return self.mask_correct_clean

class AdversarialMetrics:
    def __init__(self, context: AdversarialContext, y_pred_adv, num_breaches):
        self.ctx = context
        self.y_pred_adv = np.asarray(y_pred_adv)
        self.num_breaches = np.asarray(num_breaches)
        self.universe = self.ctx.get_universe()

    def attack_success_rate(self):
        universe = self.universe
        total = int(np.sum(universe))

        if total == 0:
            return 0.0

        if self.ctx.targeted:
            success = np.sum(
                universe & (self.y_pred_adv == self.ctx.target_class)
            )
        else:
            success = np.sum(
                universe & (self.y_pred_adv != self.ctx.y_true)
            )
        return round(100*success / total, 5)

    def adversarial_accuracy(self):
        universe = self.universe
        total = int(np.sum(universe))

        if total == 0:
            return 0.0

        if self.ctx.targeted:
            correct = np.sum(
                universe & (self.y_pred_adv != self.ctx.target_class)
            )
        else:
            correct = np.sum(
                universe & (self.y_pred_adv == self.ctx.y_true)
            )

        return round(100 * correct / total, 5)


    def get_success_mask(self):
        if self.ctx.targeted:
            return self.universe & (self.y_pred_adv == self.ctx.target_class)
        else:
            return self.universe & (self.y_pred_adv != self.ctx.y_true)
    def adversarial_accuracy_cumulative(self, max_k: int = 4):
        results = {}
        success_mask_global = self.get_success_mask()

        for k in range(max_k + 1):
            mask_k = (self.num_breaches <= k)
            mask_final = self.universe & mask_k
            total = int(mask_final.sum())

            if total == 0:
                results[k] = {
                    "acc": None, "correct": 0, "misclassified": 0,
                    "deviations": 0, "deviation_classes": None, "total": 0
                }
                continue

            misclassified = int(np.sum(mask_final & success_mask_global))
            correct = int(np.sum(mask_final & (self.y_pred_adv == self.ctx.y_true)))

            if not self.ctx.targeted:
                acc = round(100 * correct / total, 5)
                deviations = 0
                deviation_classes = None

            else:
                deviation_mask = (
                    mask_final &
                    (self.y_pred_adv != self.ctx.y_true) &
                    (self.y_pred_adv != self.ctx.target_class)
                )

                deviations = int(np.sum(deviation_mask))
                acc = round(100 * (total - misclassified) / total, 5)
                deviation_classes = None
                if deviations > 0:
                    deviation_classes = {}
                    idxs = np.where(deviation_mask)[0]
                    for i in idxs:
                        cls = int(self.y_pred_adv[i])
                        deviation_classes[cls] = deviation_classes.get(cls, 0) + 1

            results[k] = {
                "acc": acc,
                "correct": correct,
                "misclassified": misclassified,
                "deviations": deviations,
                "deviation_classes": deviation_classes,
                "total": total
            }

        return results

    def confusion_matrix_global(self, labels):
        if np.sum(self.universe) == 0:
            return None

        return confusion_matrix(
            self.ctx.y_true[self.universe],
            self.y_pred_adv[self.universe],
            labels=labels
        ).tolist()

class AdaptativeMetricV2:

    def compute_wasr_v2(self, context, validator, universe_indices, y_pred_adv, epsilon=0.05):
        N = len(universe_indices)
        if N == 0:
            return self._empty_result()

        m_mah   = validator._get_mask_mahalanobis_outliers(universe_indices)

        pen_bin = validator._get_binary_penalty_per_sample(universe_indices)
        pen_int = validator._get_integer_penalty_per_sample(universe_indices)
        pen_mah = validator._get_mahalanobis_penalty_per_sample(universe_indices)
        pen_dep, dep_logs = validator._get_dependency_penalty_v2(universe_indices, epsilon=epsilon)
        dep_ausente = dep_logs.get("dep_ausente", False)

        if dep_ausente:
            pen_total = (pen_bin + pen_int + pen_mah) / 3.0
        else:
            pen_total = 0.25 * pen_bin + 0.25 * pen_int + 0.25 * pen_dep + 0.25 * pen_mah

        pen_adj = np.minimum(pen_total, 1 - epsilon)

        y_pred_u = y_pred_adv[universe_indices]
        y_true_u = context.y_true[universe_indices]

        if context.targeted:
            success_mask = (y_pred_u == context.target_class)
        else:
            success_mask = (y_pred_u != y_true_u)

        contributions = success_mask.astype(float) * (1 - pen_adj)
        wasr_v2 = round(float(np.sum(contributions) / N), 5)

        m_bin = validator._get_mask_binarias_invalidas(universe_indices)
        m_int = validator._get_mask_inteiras_invalidas(universe_indices)
        _n_dep = (
            dep_logs.get("n_uni_violadas", np.zeros(N, dtype=int)) +
            dep_logs.get("n_excl_violadas", np.zeros(N, dtype=int)) +
            dep_logs.get("n_lin_violadas", np.zeros(N, dtype=int))
        )
        m_dep = _n_dep > 0

        k0_mask      = ~(m_bin | m_int | m_dep | m_mah)
        n_success    = int(np.sum(success_mask))
        n_k0_success = int(np.sum(success_mask & k0_mask))
        n0 = round(n_k0_success / n_success, 4) if n_success > 0 else 0.0

        num_violacoes = (
            m_bin.astype(int) + m_int.astype(int) +
            m_dep.astype(int) + m_mah.astype(int)
        )
        nk = {
            k: int(np.sum(success_mask & (num_violacoes == k)))
            for k in range(5)
        }

        constraint_breach_rate = {}
        for cname, cmask in [("binary", m_bin), ("integer", m_int),
                              ("dependency", m_dep), ("mahalanobis", m_mah)]:
            n_breached = int(np.sum(success_mask & cmask))
            constraint_breach_rate[cname] = {
                "count": n_breached,
                "rate": round(n_breached / n_success, 4) if n_success > 0 else 0.0,
            }

        realism_per_sample = 1 - pen_adj
        nk_detail = {}
        for k in range(5):
            mask_k = success_mask & (num_violacoes == k)
            n_k = int(np.sum(mask_k))
            if n_k == 0:
                nk_detail[k] = {"n": 0, "realism_medio": None, "contribution": 0.0}
            else:
                nk_detail[k] = {
                    "n": n_k,
                    "realism_medio": round(float(np.mean(realism_per_sample[mask_k])), 4),
                    "contribution": round(float(np.sum(realism_per_sample[mask_k])) / N, 4),
                }

        success_pos = np.where(success_mask)[0]
        if len(success_pos) > 0:
            penalty_percentiles = {}
            for dim, pen in [("binary", pen_bin), ("integer", pen_int),
                              ("dependency", pen_dep), ("mahalanobis", pen_mah)]:
                p25, p50, p75 = np.percentile(pen[success_pos], [25, 50, 75])
                penalty_percentiles[dim] = {
                    "p25":  round(float(p25), 4),
                    "p50":  round(float(p50), 4),
                    "mean": round(float(np.mean(pen[success_pos])), 4),
                    "p75":  round(float(p75), 4),
                }
            realism_overall = realism_per_sample[success_pos]
            r25, r50, r75 = np.percentile(realism_overall, [25, 50, 75])
            penalty_percentiles["overall_realism"] = {
                "p25": round(float(r25), 4),
                "p50": round(float(r50), 4),
                "p75": round(float(r75), 4),
            }

            dep_subtypes = [
                ("dep_uni",  "n_uni_violadas",  dep_logs.get("n_uni_total",  0)),
                ("dep_excl", "n_excl_violadas", dep_logs.get("n_excl_total", 0)),
                ("dep_lin",  "n_lin_violadas",  dep_logs.get("n_lin_total",  0)),
            ]
            penalty_percentiles["dep_subtypes"] = {}
            for key, log_key, n_total in dep_subtypes:
                if n_total == 0:
                    continue
                counts = dep_logs.get(log_key, np.zeros(N, dtype=int))
                vals = counts[success_pos].astype(float)
                p25v, p50v, p75v = np.percentile(vals, [25, 50, 75])
                penalty_percentiles["dep_subtypes"][key] = {
                    "n_total": int(n_total),
                    "p25":     round(float(p25v),          5),
                    "p50":     round(float(p50v),          5),
                    "mean":    round(float(np.mean(vals)), 5),
                    "p75":     round(float(p75v),          5),
                }
        else:
            penalty_percentiles = {}

        if len(success_pos) > 0:
            pen_mah_s = pen_mah[success_pos]
            n_s = len(success_pos)
            mah_inside = int(np.sum(pen_mah_s == 0))
            mah_mild   = int(np.sum((pen_mah_s > 0) & (pen_mah_s <= 0.1)))
            mah_severe = int(np.sum(pen_mah_s > 0.1))
            mahalanobis_zones = {
                "inside": {"count": mah_inside, "rate": round(mah_inside / n_s, 4)},
                "mild":   {"count": mah_mild,   "rate": round(mah_mild   / n_s, 4)},
                "severe": {"count": mah_severe,  "rate": round(mah_severe  / n_s, 4)},
            }
        else:
            mahalanobis_zones = {}

        return {
            "wasr_v2":      wasr_v2,
            "n0":           n0,
            "n_k0_success": n_k0_success,
            "n_success":    n_success,
            "nk":           nk,
            "n_uni_total":  dep_logs.get("n_uni_total", 0),
            "n_excl_total": dep_logs.get("n_excl_total", 0),
            "n_lin_total":  dep_logs.get("n_lin_total", 0),
            "dep_ausente":  dep_ausente,
            "pen_bin":      pen_bin,
            "pen_int":      pen_int,
            "pen_dep":      pen_dep,
            "pen_mah":      pen_mah,
            "constraint_breach_rate": constraint_breach_rate,
            "nk_detail":    nk_detail,
            "penalty_percentiles": penalty_percentiles,
            "mahalanobis_zones": mahalanobis_zones,
        }

    def _empty_result(self):
        return {
            "wasr_v2": 0.0, "n0": 0.0, "n_k0_success": 0,
            "n_success": 0, "n_uni_total": 0, "n_excl_total": 0,
            "n_lin_total": 0, "dep_ausente": True,
            "nk_detail": {}, "penalty_percentiles": {}, "constraint_breach_rate": {},
        }
