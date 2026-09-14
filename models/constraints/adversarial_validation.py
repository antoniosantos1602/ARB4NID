import numpy as np

from models.ValidationInterface import IValidationDataProvider

from models.constraints.validators.binary import BinaryValidator
from models.constraints.validators.context import ValidationContext
from models.constraints.validators.dependency import DependencyValidator
from models.constraints.validators.integer import IntegerValidator
from models.constraints.validators.mahalanobis import MahalanobisValidator


class AdversarialValidator:

    def __init__(
        self,
        X_orig,
        X_adv,
        success_indices,
        y_test,
        provider: IValidationDataProvider,
        model=None,
        y_pred_adv=None,
    ):
        self.success_indices = np.array(success_indices)

        feature_names = provider.feature_names
        feat2idx = {name: i for i, name in enumerate(feature_names)}

        constraints = provider.constraints
        self.binary_cols = constraints.binary_columns
        self.integer_cols = constraints.integer_columns
        self.dependencias_unidirecionais = constraints.unidirectional_dependencies
        self.exclusividades_mutuas = constraints.mutual_exclusivity_dependencies
        self.dependencias_lineares = constraints.linear_dependencies

        mahal_data = provider.mahalanobis_data

        self._ctx = ValidationContext(
            X_orig=X_orig,
            X_adv=X_adv,
            success_indices=self.success_indices,
            y_test=y_test,
            y_pred_adv=y_pred_adv,
            model=model,
            feature_names=feature_names,
            feat2idx=feat2idx,
            binary_cols=self.binary_cols,
            integer_cols=self.integer_cols,
            unidirectional_dependencies=self.dependencias_unidirecionais,
            mutual_exclusivity_dependencies=self.exclusividades_mutuas,
            linear_dependencies=self.dependencias_lineares,
            mahalanobis_refs=mahal_data["refs"],
            mahal_flags_refs=mahal_data["flags_refs"],
            mahal_label_refs=mahal_data["label_refs"],
            mahal_global_ref=mahal_data["global_ref"],
            indices_mahalanobis=mahal_data["idxs"],
            variaveis_mahalanobis=mahal_data["vars"],
            grupos_invalidos=mahal_data["grupos_invalidos"],
        )

        self._binary = BinaryValidator(self._ctx)
        self._integer = IntegerValidator(self._ctx)
        self._dependency = DependencyValidator(self._ctx)
        self._mahalanobis = MahalanobisValidator(self._ctx, self._dependency)


    def _get_mask_binarias_invalidas(self, indices=None):
        indices = self.success_indices if indices is None else indices
        return self._binary.mask_invalid(indices)

    def _get_mask_inteiras_invalidas(self, indices=None):
        indices = self.success_indices if indices is None else indices
        return self._integer.mask_invalid(indices)

    def _get_mask_mahalanobis_outliers(self, indices=None):
        indices = self.success_indices if indices is None else indices
        return self._mahalanobis.mask_outliers(indices)


    def _get_binary_penalty_per_sample(self, indices=None):
        indices = self.success_indices if indices is None else indices
        return self._binary.penalty_per_sample(indices)

    def _get_integer_penalty_per_sample(self, indices=None):
        indices = self.success_indices if indices is None else indices
        return self._integer.penalty_per_sample(indices)

    def _get_dependency_penalty_v2(self, indices=None, epsilon=0.05):
        indices = self.success_indices if indices is None else indices
        return self._dependency.penalty_v2(indices, epsilon=epsilon)

    def _get_mahalanobis_penalty_per_sample(self, indices=None):
        indices = self.success_indices if indices is None else indices
        return self._mahalanobis.penalty_per_sample(indices)


    def _count_uni_violations(self, indices):
        return self._dependency.count_unidirectional(indices)

    def _count_excl_violations(self, indices):
        return self._dependency.count_mutual_exclusivity(indices)

    def _count_lin_violations(self, indices):
        return self._dependency.count_linear(indices)


    def get_detailed_broken_rules(self, indices=None):
        indices = self.success_indices if indices is None else indices
        return self._dependency.detailed_broken_rules(indices)
