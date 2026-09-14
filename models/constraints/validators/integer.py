import numpy as np

from models.constraints.validators.context import ValidationContext


class IntegerValidator:

    def __init__(self, ctx: ValidationContext) -> None:
        self.ctx = ctx

    def mask_invalid(self, indices) -> np.ndarray:
        ctx = self.ctx
        if not ctx.integer_cols:
            return np.zeros(len(indices), dtype=bool)

        integer_indices = [ctx.feat2idx[c] for c in ctx.integer_cols if c in ctx.feat2idx]
        X_orig_int = ctx.X_orig.iloc[indices][ctx.integer_cols].values
        X_adv_int = ctx.X_adv[indices][:, integer_indices]

        alteradas = X_orig_int != X_adv_int
        invalidas = (X_adv_int < 0) | (~np.isclose(X_adv_int, np.round(X_adv_int)))
        return np.any(alteradas & invalidas, axis=1)

    def penalty_per_sample(self, indices) -> np.ndarray:
        ctx = self.ctx
        if not ctx.integer_cols:
            return np.zeros(len(indices), dtype=float)

        integer_idxs = [ctx.feat2idx[c] for c in ctx.integer_cols if c in ctx.feat2idx]
        total_ints = len(integer_idxs)

        X_orig_int = ctx.X_orig.iloc[indices][ctx.integer_cols].values
        X_adv_int = ctx.X_adv[indices][:, integer_idxs]

        alteradas = X_orig_int != X_adv_int
        invalidas = (X_adv_int < 0) | (~np.isclose(X_adv_int, np.round(X_adv_int)))
        violadas_por_feature = alteradas & invalidas

        return np.sum(violadas_por_feature, axis=1) / total_ints
