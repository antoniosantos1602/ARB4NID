import numpy as np

from models.constraints.validators.context import ValidationContext
from models.constraints.validators.dependency import DependencyValidator
from utils.math_utils import safe_mahalanobis


class MahalanobisValidator:
    def __init__(self, ctx: ValidationContext, dependency: DependencyValidator) -> None:
        self.ctx = ctx
        self.dependency = dependency
        self.meu_log_mahalanobis: list = []


    def _label_plus_flags_ref(self, label_value, fkey):
        ctx = self.ctx
        key = tuple(sorted((('label', label_value),) + tuple(fkey)))
        ref = ctx.mahalanobis_refs.get(key)
        return ref if (ref and key not in ctx.grupos_invalidos) else None


    def mask_outliers(self, indices) -> np.ndarray:
        ctx = self.ctx
        mask = np.zeros(len(indices), dtype=bool)
        self.meu_log_mahalanobis = []

        for i, idx in enumerate(indices):
            x = ctx.X_adv[idx][ctx.indices_mahalanobis]
            y_pred = ctx.y_pred_adv[idx] if ctx.y_pred_adv is not None else None

            flags_vals = {c: ctx.X_adv[idx][ctx.feat2idx[c]] for c in ctx.binary_cols}
            has_invalid_binarias = any(
                (not np.isclose(v, 0)) and (not np.isclose(v, 1))
                for v in flags_vals.values()
            )
            mask_dep_exc = self.dependency.mask_any([idx])

            if mask_dep_exc[0] or has_invalid_binarias:
                fkey = tuple()
            else:
                fkey = tuple(sorted([(c, 1) for c in flags_vals if np.isclose(flags_vals[c], 1)]))

            ref = None
            tipo = ""

            if y_pred is not None and len(fkey) > 0:
                ref = self._label_plus_flags_ref(y_pred, fkey)
                tipo = "label+flags"

            if ref is None and y_pred is not None:
                ref = ctx.mahal_label_refs.get(y_pred)
                tipo = "label_forced" if (mask_dep_exc[0] or has_invalid_binarias) else "label"

            if ref is None:
                ref = ctx.mahal_global_ref
                tipo = "global"

            if ref:
                d = safe_mahalanobis(x, ref["mean"], ref["cov_inv"])
                thr = ref["threshold"]
                is_outlier = bool(d > thr)
                mask[i] = is_outlier

                self.meu_log_mahalanobis.append({
                    "idx": int(idx),
                    "tipo_grupo": tipo,
                    "distancia": round(float(d), 4),
                    "threshold": round(float(thr), 4),
                    "outlier": is_outlier,
                    "flags_ativas": str(fkey),
                })
            else:
                mask[i] = True
                self.meu_log_mahalanobis.append({
                    "idx": int(idx),
                    "tipo_grupo": "sem_grupo",
                    "distancia": None,
                    "threshold": None,
                    "outlier": True,
                    "flags_ativas": str(fkey),
                })

        return mask

    def penalty_per_sample(self, indices) -> np.ndarray:
        penalties = np.zeros(len(indices), dtype=float)
        log_por_idx = {e["idx"]: e for e in self.meu_log_mahalanobis}

        for i, idx in enumerate(indices):
            entry = log_por_idx.get(int(idx))

            if (
                entry is None
                or entry["distancia"] is None
                or entry["threshold"] is None
                or entry["threshold"] == 0
            ):
                penalties[i] = 1.0
                continue

            r = entry["distancia"] / entry["threshold"]

            if r <= 1.0:
                penalties[i] = 0.0
            elif r <= 2.0:
                penalties[i] = 0.1 * (r - 1.0)
            else:
                penalties[i] = 0.1 + 0.9 * (1 - 2 / r)

        return penalties
