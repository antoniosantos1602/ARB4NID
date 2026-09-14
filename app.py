from flask import Flask
from flask_cors import CORS
import warnings

from controllers.routes.routes import api_bp
from controllers.routes.views import views_bp
from controllers.routes.errors import register_error_handlers
from models.container import register_services

warnings.simplefilter("always", category=FutureWarning)

app = Flask(__name__, template_folder='views/templates', static_folder='views/static')
app.secret_key = 'supersecretkey'
CORS(app, resources={r"/api/*": {"origins": "*"}})

app.register_blueprint(views_bp)
app.register_blueprint(api_bp)
register_services(app)
register_error_handlers(app)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    app.run(debug=True)
