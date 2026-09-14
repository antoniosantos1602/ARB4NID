from abc import ABC, abstractmethod
from models.constraints.executar_ataque import executar_ataque_realista

class BaseAttack(ABC):

    def __init__(self, name, tool):
        self.name = name
        self.tool = tool

    @abstractmethod
    def get_params(self, form):
        pass

    @abstractmethod
    def generate_x_adv(self, model, X_test, y_test, config):
        pass

    def run(self, form, model_paths, shared_params):
        config = self.get_params(form)

        return executar_ataque_realista(
            attack_fn=self.generate_x_adv,
            attack_params={
                "config": config,
                "attack_name": self.name,
                "attack_tool": self.tool
            },
            model_paths=model_paths,
            X_data=shared_params["X_test"],
            y_data=shared_params["y_test"],
            df_data_original=shared_params["df_original"],
            column_names=shared_params["column_names"],
            intelligence=shared_params["intelligence"],
        )
