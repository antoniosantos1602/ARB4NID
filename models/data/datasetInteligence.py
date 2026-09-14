import numpy as np
from utils.math_utils import safe_mahalanobis
from itertools import combinations
from models.ValidationInterface import IValidationDataProvider
from models.data.constraints_discover import discover_all_constraints


class DatasetIntelligence(IValidationDataProvider):

    def __init__(self, df_original):
        self._df_orig = df_original.copy()
        self._feature_names = [c for c in df_original.columns if c != 'label']
        self._feat2idx = {name: i for i, name in enumerate(self._feature_names)}

        self._constraints = discover_all_constraints(df_original.drop(columns=['label'], errors='ignore'))

        self.binary_cols = self._constraints.binary_columns

        self._mahal_cache = self._build_mahalanobis_reference()
        print("DatasetIntelligence")


    @property
    def constraints(self): return self._constraints

    @property
    def mahalanobis_data(self): return self._mahal_cache

    @property
    def feature_names(self): return self._feature_names

    def _get_all_binary_groupings(self, row):
        base = [('label', row['label'])]
        active_bins = [(col, row[col]) for col in self.binary_cols if row.get(col, 0) == 1]
        for i in range(len(active_bins), -1, -1):
            for comb in combinations(active_bins, i):
                yield tuple(sorted(base + list(comb)))

    def _build_mahalanobis_reference(self):
        X_orig = self._df_orig

        variaveis_mahalanobis = [
            col for col in X_orig.columns
            if col not in self.binary_cols and col != 'label'
        ]
        indices_mahalanobis = [self._feat2idx[col] for col in variaveis_mahalanobis]

        mahalanobis_refs = {}
        grupos_invalidos = set()

        for idx, row in X_orig.iterrows():
            for grupo in self._get_all_binary_groupings(row):
                mahalanobis_refs.setdefault(grupo, []).append(idx)

        for grupo, indices in mahalanobis_refs.items():
            if len(indices) < len(variaveis_mahalanobis):
                grupos_invalidos.add(grupo)
                continue
            arr = X_orig.iloc[indices][variaveis_mahalanobis].values
            mean = np.mean(arr, axis=0)
            cov = np.cov(arr, rowvar=False) + 1e-6 * np.eye(arr.shape[1])
            cov_inv = np.linalg.pinv(cov)
            threshold = np.percentile([safe_mahalanobis(x, mean, cov_inv) for x in arr], 99.5)

            mahalanobis_refs[grupo] = {
                "mean": mean, "cov_inv": cov_inv, "threshold": threshold
            }

        mahal_flags_refs = {}
        grupos_flags_invalidos = set()

        def _flags_key_from_row(row):
            return tuple(sorted([(c, 1) for c in self.binary_cols if row.get(c, 0) == 1]))

        flags_to_indices = {}
        for idx, row in X_orig.iterrows():
            fkey = _flags_key_from_row(row)
            flags_to_indices.setdefault(fkey, []).append(idx)

        for fkey, indices in flags_to_indices.items():
            if len(indices) < len(variaveis_mahalanobis):
                grupos_flags_invalidos.add(fkey)
                continue
            arr = X_orig.iloc[indices][variaveis_mahalanobis].values
            mean = np.mean(arr, axis=0)
            cov = np.cov(arr, rowvar=False) + 1e-6 * np.eye(arr.shape[1])
            cov_inv = np.linalg.pinv(cov)
            threshold = np.percentile([safe_mahalanobis(x, mean, cov_inv) for x in arr], 99.5)

            mahal_flags_refs[fkey] = {
                "mean": mean, "cov_inv": cov_inv, "threshold": threshold
            }

        mahal_label_refs = {}
        for label_value in X_orig['label'].unique():
            arr = X_orig[X_orig['label'] == label_value][variaveis_mahalanobis].values
            if len(arr) < len(variaveis_mahalanobis):
                continue
            mean = np.mean(arr, axis=0)
            cov = np.cov(arr, rowvar=False) + 1e-6 * np.eye(arr.shape[1])
            cov_inv = np.linalg.pinv(cov)
            threshold = np.percentile([safe_mahalanobis(x, mean, cov_inv) for x in arr], 99.5)

            mahal_label_refs[label_value] = {
                "mean": mean, "cov_inv": cov_inv, "threshold": threshold
            }

        arr_all = X_orig[variaveis_mahalanobis].values
        mahal_global_ref = None
        if len(arr_all) >= len(variaveis_mahalanobis):
            mean = np.mean(arr_all, axis=0)
            cov = np.cov(arr_all, rowvar=False) + 1e-6 * np.eye(arr_all.shape[1])
            cov_inv = np.linalg.pinv(cov)
            threshold = np.percentile([safe_mahalanobis(x, mean, cov_inv) for x in arr_all], 99.5)
            mahal_global_ref = {"mean": mean, "cov_inv": cov_inv, "threshold": threshold}

        return {
            "refs": mahalanobis_refs,
            "flags_refs": mahal_flags_refs,
            "label_refs": mahal_label_refs,
            "global_ref": mahal_global_ref,
            "vars": variaveis_mahalanobis,
            "idxs": indices_mahalanobis,
            "grupos_invalidos": grupos_invalidos,
            "grupos_flags_invalidos": grupos_flags_invalidos
        }
