from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class ValidationContext:
    X_orig: pd.DataFrame
    X_adv: np.ndarray
    success_indices: np.ndarray
    y_test: np.ndarray
    y_pred_adv: Optional[np.ndarray] = None
    model: Optional[Any] = None

    feature_names: list = field(default_factory=list)
    feat2idx: dict = field(default_factory=dict)

    binary_cols: list = field(default_factory=list)
    integer_cols: list = field(default_factory=list)
    unidirectional_dependencies: list = field(default_factory=list)
    mutual_exclusivity_dependencies: list = field(default_factory=list)
    linear_dependencies: list = field(default_factory=list)

    mahalanobis_refs: dict = field(default_factory=dict)
    mahal_flags_refs: dict = field(default_factory=dict)
    mahal_label_refs: dict = field(default_factory=dict)
    mahal_global_ref: Optional[dict] = None
    indices_mahalanobis: list = field(default_factory=list)
    variaveis_mahalanobis: list = field(default_factory=list)
    grupos_invalidos: Any = None
