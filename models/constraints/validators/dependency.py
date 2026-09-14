import numpy as np

from models.constraints.validators.context import ValidationContext


class DependencyValidator:
    def __init__(self, ctx: ValidationContext) -> None:
        self.ctx = ctx


    def _x_adv_disc(self, indices) -> np.ndarray:
        ctx = self.ctx
        bin_idxs = [ctx.feat2idx[c] for c in ctx.binary_cols if c in ctx.feat2idx]
        X_disc = ctx.X_adv[indices].copy()
        if bin_idxs:
            X_disc[:, bin_idxs] = np.round(X_disc[:, bin_idxs]).clip(0, 1)
        return X_disc


    def mask_unidirectional(self, indices) -> np.ndarray:
        ctx = self.ctx
        mask = np.zeros(len(indices), dtype=bool)
        for A, B, _ in ctx.unidirectional_dependencies:
            if A in ctx.feature_names and B in ctx.feature_names:
                idx_A, idx_B = ctx.feat2idx[A], ctx.feat2idx[B]
                mask |= (
                    np.isclose(ctx.X_adv[indices][:, idx_A], 1)
                    & np.isclose(ctx.X_adv[indices][:, idx_B], 0)
                )
        return mask

    def mask_mutual_exclusivity(self, indices) -> np.ndarray:
        ctx = self.ctx
        mask = np.zeros(len(indices), dtype=bool)
        for col1, col2 in ctx.mutual_exclusivity_dependencies:
            if col1 in ctx.feature_names and col2 in ctx.feature_names:
                idx1, idx2 = ctx.feat2idx[col1], ctx.feat2idx[col2]
                mask |= (
                    np.isclose(ctx.X_adv[indices][:, idx1], 1)
                    & np.isclose(ctx.X_adv[indices][:, idx2], 1)
                )
        return mask

    def mask_linear(self, indices, tol: float = 1e-6) -> np.ndarray:
        ctx = self.ctx
        mask = np.zeros(len(indices), dtype=bool)
        for target, sources in ctx.linear_dependencies:
            idx_t = ctx.feat2idx[target]
            idx_s = [ctx.feat2idx[s] for s in sources]
            vals_sum = np.sum(ctx.X_adv[indices][:, idx_s], axis=1)
            mask |= ~np.isclose(ctx.X_adv[indices][:, idx_t], vals_sum, atol=tol)
        return mask

    def mask_any(self, indices) -> np.ndarray:
        return self.mask_unidirectional(indices) | self.mask_mutual_exclusivity(indices) | self.mask_linear(indices)


    def count_unidirectional(self, indices) -> np.ndarray:
        ctx = self.ctx
        X_disc = self._x_adv_disc(indices)
        counts = np.zeros(len(indices), dtype=int)
        for A, B, _ in ctx.unidirectional_dependencies:
            if A in ctx.feature_names and B in ctx.feature_names:
                idx_A, idx_B = ctx.feat2idx[A], ctx.feat2idx[B]
                counts += (
                    np.isclose(X_disc[:, idx_A], 1)
                    & np.isclose(X_disc[:, idx_B], 0)
                ).astype(int)
        return counts

    def count_mutual_exclusivity(self, indices) -> np.ndarray:
        ctx = self.ctx
        X_disc = self._x_adv_disc(indices)
        counts = np.zeros(len(indices), dtype=int)
        for col1, col2 in ctx.mutual_exclusivity_dependencies:
            if col1 in ctx.feature_names and col2 in ctx.feature_names:
                idx1, idx2 = ctx.feat2idx[col1], ctx.feat2idx[col2]
                counts += (
                    np.isclose(X_disc[:, idx1], 1)
                    & np.isclose(X_disc[:, idx2], 1)
                ).astype(int)
        return counts

    def count_linear(self, indices) -> np.ndarray:
        ctx = self.ctx
        X_disc = self._x_adv_disc(indices)
        counts = np.zeros(len(indices), dtype=int)
        for target, sources in ctx.linear_dependencies:
            idx_t = ctx.feat2idx[target]
            idx_s = [ctx.feat2idx[s] for s in sources]
            vals_sum = np.sum(X_disc[:, idx_s], axis=1)
            counts += (~np.isclose(X_disc[:, idx_t], vals_sum, atol=1e-6)).astype(int)
        return counts


    def penalty_v2(self, indices, epsilon: float = 0.05):
        ctx = self.ctx
        n = len(indices)

        n_uni_total = len(ctx.unidirectional_dependencies)
        n_excl_total = len(ctx.mutual_exclusivity_dependencies)
        n_lin_total = len(ctx.linear_dependencies)

        logs = {
            "n_uni_total":  n_uni_total,
            "n_excl_total": n_excl_total,
            "n_lin_total":  n_lin_total,
        }

        scores = []

        if n_uni_total > 0:
            counts = self.count_unidirectional(indices)
            pen = np.log(1 + counts) / np.log(1 + n_uni_total)
            scores.append(np.maximum(1 - pen, epsilon))
            logs["n_uni_violadas"] = counts
            logs["pen_uni"] = pen
        else:
            logs["n_uni_violadas"] = np.zeros(n, dtype=int)
            logs["pen_uni"] = np.zeros(n, dtype=float)

        if n_excl_total > 0:
            counts = self.count_mutual_exclusivity(indices)
            pen = np.log(1 + counts) / np.log(1 + n_excl_total)
            scores.append(np.maximum(1 - pen, epsilon))
            logs["n_excl_violadas"] = counts
            logs["pen_excl"] = pen
        else:
            logs["n_excl_violadas"] = np.zeros(n, dtype=int)
            logs["pen_excl"] = np.zeros(n, dtype=float)

        if n_lin_total > 0:
            counts = self.count_linear(indices)
            pen = np.log(1 + counts) / np.log(1 + n_lin_total)
            scores.append(np.maximum(1 - pen, epsilon))
            logs["n_lin_violadas"] = counts
            logs["pen_lin"] = pen
        else:
            logs["n_lin_violadas"] = np.zeros(n, dtype=int)
            logs["pen_lin"] = np.zeros(n, dtype=float)

        if not scores:
            logs["pen_dep_v2"] = np.zeros(n, dtype=float)
            logs["dep_ausente"] = True
            return np.zeros(n, dtype=float), logs

        K = len(scores)
        H = K / sum(1.0 / s for s in scores)
        pen_dep_v2 = 1 - H

        logs["pen_dep_v2"] = pen_dep_v2
        logs["dep_ausente"] = False
        return pen_dep_v2, logs


    def detailed_broken_rules(self, indices) -> dict:
        ctx = self.ctx
        detailed_report = {idx: [] for idx in indices}
        X_disc = self._x_adv_disc(indices)

        for A, B, _ in ctx.unidirectional_dependencies:
            if A in ctx.feature_names and B in ctx.feature_names:
                idx_A, idx_B = ctx.feat2idx[A], ctx.feat2idx[B]
                violacoes = (
                    np.isclose(X_disc[:, idx_A], 1)
                    & np.isclose(X_disc[:, idx_B], 0)
                )
                for i, violou in enumerate(violacoes):
                    if violou:
                        detailed_report[indices[i]].append(f"Unidirectional: {A} require {B}")

        for col1, col2 in ctx.mutual_exclusivity_dependencies:
            if col1 in ctx.feature_names and col2 in ctx.feature_names:
                idx1, idx2 = ctx.feat2idx[col1], ctx.feat2idx[col2]
                violacoes = (
                    np.isclose(X_disc[:, idx1], 1)
                    & np.isclose(X_disc[:, idx2], 1)
                )
                for i, violou in enumerate(violacoes):
                    if violou:
                        detailed_report[indices[i]].append(f"Mutual Exclusivity: {col1} e {col2}")

        for target, sources in ctx.linear_dependencies:
            idx_t = ctx.feat2idx[target]
            idx_s = [ctx.feat2idx[s] for s in sources]
            vals_sum = np.sum(X_disc[:, idx_s], axis=1)
            violacoes = ~np.isclose(X_disc[:, idx_t], vals_sum, atol=1e-6)
            for i, violou in enumerate(violacoes):
                if violou:
                    detailed_report[indices[i]].append(f"{target} it must be the sum of {'+'.join(sources)}")

        return detailed_report
