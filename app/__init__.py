from flask import Flask
from app.routes_folder.evasion_routes import evasion
from app.routes_folder.poisoning_routes import poisoning

def create_app():
    app = Flask(
        __name__,
        template_folder='../templates',  # Caminho relativo à raiz do projeto
        static_folder='../static'
    )

    from .routes import main
    app.register_blueprint(main)
    app.register_blueprint(evasion)
    app.register_blueprint(poisoning)

    return app
