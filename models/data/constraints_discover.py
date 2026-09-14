from models.data import feature_utils
import pandas as pd
from typing import List, Tuple
import numpy as np

class ConstraintSet:
    def __init__(self, unidirectional_dependencies, mutual_exclusivity_dependencies , linear_dependencies, integer_columns, binary_columns):
        self.unidirectional_dependencies = unidirectional_dependencies
        self.mutual_exclusivity_dependencies = mutual_exclusivity_dependencies
        self.linear_dependencies = linear_dependencies
        self.integer_columns = integer_columns
        self.binary_columns = binary_columns

        self.total_dependencies = (
            len(unidirectional_dependencies) +
            len(mutual_exclusivity_dependencies) +
            len(linear_dependencies)
        )


def find_unidirectional_dependencies(df: pd.DataFrame, threshold: float = 1) -> List[Tuple]:
    bin_cols = feature_utils.get_binary_columns(df)
    dependencies = []

    for colA in bin_cols:
        for colB in bin_cols:
            if colA == colB: continue
            cond = df[colA] == 1
            if cond.sum() == 0: continue
            prob = df.loc[cond, colB].mean()
            if prob >= threshold:
                dependencies.append((colA, colB, round(prob, 3)))

    return dependencies

def find_mutual_exclusivity_dependencies(df: pd.DataFrame) -> List[Tuple]:
    bin_cols = feature_utils.get_binary_columns(df)
    exclusivities = []

    for i in range(len(bin_cols)):
        for j in range(i + 1, len(bin_cols)):
            col1 = bin_cols[i]
            col2 = bin_cols[j]
            ambos_ativos = (df[col1] == 1) & (df[col2] == 1)
            if not ambos_ativos.any():
                exclusivities.append((col1, col2))

    return exclusivities

def find_linear_dependencies(df: pd.DataFrame, tol_corr: float = 0.999, tol_dif: float = 1e-6, min_unique: int = 20) -> Tuple[List[Tuple], List[str]]:
    num_cols = df.select_dtypes(include='number').columns

    continuas = []
    for col in num_cols:
        unique_vals = df[col].nunique(dropna=True)
        if unique_vals <= 2 or unique_vals < min_unique: continue
        continuas.append(col)

    relacoes = []
    for target in continuas:
        for c1 in continuas:
            for c2 in continuas:
                if len({target, c1, c2}) < 3: continue

                soma = df[c1] + df[c2]
                corr = np.corrcoef(df[target], soma)[0, 1]
                diffs = (df[target] - soma).abs()

                if corr >= tol_corr and (diffs < tol_dif).mean() > 0.99:
                    relacoes.append((target, [c1, c2]))

    return relacoes, continuas

def discover_all_constraints (df: pd.DataFrame, threshold: float = 1.0):
    integer_columns = feature_utils.get_integer_columns(df)
    binary_columns = feature_utils.get_binary_columns(df)

    unidirectional_dependencies = find_unidirectional_dependencies(df, threshold=threshold)
    mutual_exclusivity_dependencies = find_mutual_exclusivity_dependencies(df)
    linear_dependencies, _ = find_linear_dependencies(df)

    return ConstraintSet(
        unidirectional_dependencies=unidirectional_dependencies,
        mutual_exclusivity_dependencies=mutual_exclusivity_dependencies,
        linear_dependencies=linear_dependencies,
        integer_columns=integer_columns,
        binary_columns=binary_columns
    )
