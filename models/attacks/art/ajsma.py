import numpy as np
import pandas as pd
from dataclasses import dataclass

from utils.wrapper import create_art_classifier
from models.attacks.base_attack import BaseAttack


@dataclass
class AJSMAConfig:
    theta: float
    gamma: float
    targeted: bool
    target_class: int
    n_controllable: int
    respect_constraints: bool
    seed: int


class AJSMAAttack(BaseAttack):
    def __init__(self):
        super().__init__(name="AJSMA", tool="ART")

    def get_params(self, form) -> AJSMAConfig:
        try:
            theta_raw = form.get("ajsma_theta_manual") or form.get("ajsma_theta")
            gamma_raw = form.get("ajsma_gamma_manual") or form.get("ajsma_gamma")
            n_ctrl_raw = form.get("ajsma_n_controllable", 0)
            target_class_raw = form.get("ajsma_target_class", 0)

            return AJSMAConfig(
                theta=float(theta_raw or 0.1),
                gamma=float(gamma_raw or 0.1),
                targeted=form.get("ajsma_targeted") == "True",
                target_class=int(target_class_raw),
                n_controllable=int(n_ctrl_raw or 0),
                respect_constraints=form.get("ajsma_respect_constraints", "True") != "False",
                seed=int(form.get("ajsma_seed", 0) or 0),
            )
        except ValueError:
            raise ValueError("Invalid Parameters")

    def generate_x_adv(self, model, X_test, y_test, config: AJSMAConfig,
                       intelligence=None, scaler=None, column_names=None, **kwargs):
        try:
            art_classifier = create_art_classifier(model, X_test, y_test, attack_type="AJSMA")

            feature_names = list(column_names) if column_names is not None else (
                list(intelligence.feature_names) if intelligence is not None else None
            )
            constraints = intelligence.constraints if (
                intelligence is not None and config.respect_constraints
            ) else None

            engine = _AJSMAEngine(
                clf=art_classifier,
                scaler=scaler,
                feature_names=feature_names,
                constraints=constraints,
                cfg=config,
                X_scaled=np.asarray(X_test, dtype=float),
                y_true=np.asarray(y_test).astype(int),
            )
            return engine.run()
        except Exception:
            raise TypeError("This model is not supported for this type of attack.")


class _AJSMAEngine:
    def __init__(self, clf, scaler, feature_names, constraints, cfg, X_scaled, y_true):
        self.clf = clf
        self.scaler = scaler
        self.cfg = cfg
        self.y_true = y_true
        self.X_scaled = X_scaled
        self.n, self.F = X_scaled.shape

        self.x_orig = self._to_orig(X_scaled)
        self.feat_std = np.std(self.x_orig, axis=0)
        self.feat_std[self.feat_std < 1e-8] = 1.0
        self.feat_min = self.x_orig.min(axis=0)
        self.feat_max = self.x_orig.max(axis=0)

        self._build_constraints(constraints, feature_names)


    def _build_constraints(self, cset, feature_names):
        self.bin_idx, self.int_idx = [], []
        self.mutex_groups, self.uni_pairs, self.linear_rules = [], [], []
        self.linear_targets = set()

        if cset is None or feature_names is None:
            return

        name2idx = {name: i for i, name in enumerate(feature_names)}

        def idx(name):
            return name2idx.get(name)

        self.bin_idx = [idx(c) for c in cset.binary_columns if idx(c) is not None]
        self.int_idx = [idx(c) for c in cset.integer_columns if idx(c) is not None]

        mutex_pairs = []
        universe = set()
        for c1, c2 in cset.mutual_exclusivity_dependencies:
            i, j = idx(c1), idx(c2)
            if i is not None and j is not None:
                mutex_pairs.append((i, j))
                universe.update((i, j))
        self.mutex_groups = _connected_components(mutex_pairs, universe)

        for a, b, *_ in cset.unidirectional_dependencies:
            ia, ib = idx(a), idx(b)
            if ia is not None and ib is not None:
                self.uni_pairs.append((ia, ib))

        for target, sources in cset.linear_dependencies:
            it = idx(target)
            iss = [idx(s) for s in sources]
            if it is not None and all(s is not None for s in iss):
                self.linear_rules.append((it, iss))
                self.linear_targets.add(it)


    def _to_orig(self, x_scaled):
        if self.scaler is None:
            return np.array(x_scaled, dtype=float, copy=True)
        return np.asarray(self.scaler.inverse_transform(x_scaled), dtype=float)

    def _to_scaled(self, x_orig):
        if self.scaler is None:
            return np.asarray(x_orig, dtype=np.float32)
        df = pd.DataFrame(x_orig, columns=self.scaler.feature_names_in_)
        return np.asarray(self.scaler.transform(df), dtype=np.float32)

    def _predict(self, x_orig):
        probs = self.clf.predict(self._to_scaled(x_orig))
        return np.argmax(probs, axis=1)


    def _resolve_targets(self):
        if self.cfg.targeted:
            return np.full(self.n, self.cfg.target_class, dtype=int)
        probs = self.clf.predict(self.X_scaled)
        probs = probs.copy()
        probs[np.arange(self.n), self.y_true] = -np.inf
        return np.argmax(probs, axis=1)

    def _controllable_mask(self):
        mask = np.ones(self.F, dtype=bool)
        for t in self.linear_targets:
            mask[t] = False
        if self.cfg.n_controllable and self.cfg.n_controllable > 0:
            pool = np.where(mask)[0]
            k = min(self.cfg.n_controllable, len(pool))
            rng = np.random.default_rng(self.cfg.seed)
            chosen = set(rng.choice(pool, size=k, replace=False).tolist())
            mask = np.zeros(self.F, dtype=bool)
            for c in chosen:
                mask[c] = True
        return mask

    def _reached(self, preds):
        if self.cfg.targeted:
            return preds == self.cfg.target_class
        return preds != self.y_true


    def run(self):
        x = self.x_orig
        orig_ref = x.copy()
        grad_target = self._resolve_targets()
        controllable = self._controllable_mask()
        bin_set = set(self.bin_idx)

        budget = max(1, int(np.ceil(self.cfg.gamma * self.F)))
        max_iter = budget + 5

        active = ~self._reached(self._predict(x))
        it = 0
        while active.any() and it < max_iter:
            it += 1
            idx_active = np.where(active)[0]

            grads = self.clf.class_gradient(self._to_scaled(x[idx_active]))
            best_feat, best_dir, has_cand = self._select(
                grads, grad_target[idx_active], x[idx_active], controllable, bin_set
            )

            for k, si in enumerate(idx_active):
                if not has_cand[k]:
                    active[si] = False
                    continue
                self._apply(x[si], best_feat[k], best_dir[k], bin_set)
                self._repair(x[si])
                changed = np.count_nonzero(~np.isclose(x[si], orig_ref[si]))
                if changed >= budget:
                    active[si] = False

            active &= ~self._reached(self._predict(x))

        return self._to_scaled(x)


    def _select(self, grads, targets, x_active, controllable, bin_set):
        na = grads.shape[0]
        alpha = grads[np.arange(na), targets, :]
        beta = grads.sum(axis=1) - alpha

        inc = (alpha > 0) & (beta < 0)
        dec = (alpha < 0) & (beta > 0)
        s_inc = np.where(inc, alpha * (-beta), -np.inf)
        s_dec = np.where(dec, (-alpha) * beta, -np.inf)
        score = np.maximum(s_inc, s_dec)
        direction = np.where(s_inc >= s_dec, 1, -1)

        feasible = np.broadcast_to(controllable, (na, self.F)).copy()

        if bin_set:
            b = np.array(sorted(bin_set))
            at_one = x_active[:, b] > 0.5
            up = direction[:, b] > 0
            waste = (up & at_one) | (~up & ~at_one)
            feasible[:, b] &= ~waste

        score = np.where(feasible, score, -np.inf)

        best_feat = np.argmax(score, axis=1)
        has_cand = score[np.arange(na), best_feat] > -np.inf
        best_dir = direction[np.arange(na), best_feat]
        return best_feat, best_dir, has_cand


    def _apply(self, x, i, d, bin_set):
        if i in bin_set:
            x[i] = 1.0 if d > 0 else 0.0
        elif i in self.int_idx:
            step = max(1, int(round(self.cfg.theta * self.feat_std[i])))
            x[i] = max(0.0, round(x[i] + d * step))
        else:
            x[i] = np.clip(
                x[i] + d * self.cfg.theta * self.feat_std[i],
                self.feat_min[i], self.feat_max[i],
            )


    def _repair(self, x):
        for group in self.mutex_groups:
            actives = [g for g in group if x[g] > 0.5]
            if len(actives) > 1:
                keep = max(actives, key=lambda g: x[g])
                for g in group:
                    if g != keep:
                        x[g] = 0.0

        for a, b in self.uni_pairs:
            if x[a] > 0.5 and x[b] < 0.5:
                x[b] = 1.0

        for t, sources in self.linear_rules:
            x[t] = float(np.sum([x[s] for s in sources]))


def _connected_components(pairs, universe):
    parent = {u: u for u in universe}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        parent[find(a)] = find(b)

    groups = {}
    for u in universe:
        groups.setdefault(find(u), []).append(u)
    return [sorted(g) for g in groups.values() if len(g) > 1]
