import numpy as np

from models.constraints.validators.context import ValidationContext


class BinaryValidator:

    def __init__(self, ctx: ValidationContext) -> None:
        self.ctx = ctx

    def mask_invalid(self, indices) -> np.ndarray:
        ctx = self.ctx
        if not ctx.binary_cols:
            return np.zeros(len(indices), dtype=bool)

        bin_idxs = [ctx.feat2idx[c] for c in ctx.binary_cols if c in ctx.feat2idx]
        Xb = ctx.X_adv[indices][:, bin_idxs]

        is_01 = np.isin(Xb, [0, 1]) & (Xb == Xb.astype(int))
        invalid_values_mask = ~np.all(is_01, axis=1)
        all_zero_mask = np.all(np.isclose(Xb, 0), axis=1)
        return invalid_values_mask | all_zero_mask

    def penalty_per_sample(self, indices) -> np.ndarray:
        ctx = self.ctx
        if not ctx.binary_cols:
            return np.zeros(len(indices), dtype=float)

        bin_idxs = [ctx.feat2idx[c] for c in ctx.binary_cols if c in ctx.feat2idx]
        total_bins = len(bin_idxs)

        Xb = ctx.X_adv[indices][:, bin_idxs]
        is_01 = np.isin(Xb, [0, 1]) & (Xb == Xb.astype(int))
        invalidas_por_feature = ~is_01

        return np.sum(invalidas_por_feature, axis=1) / total_bins
