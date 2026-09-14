from abc import ABC, abstractmethod

class IValidationDataProvider(ABC):

    @property
    @abstractmethod
    def constraints(self):
        pass

    @property
    @abstractmethod
    def mahalanobis_data(self):
        pass

    @property
    @abstractmethod
    def feature_names(self):
        pass
