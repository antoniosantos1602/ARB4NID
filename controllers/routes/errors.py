from flask import jsonify


def _err(message, code, details=None):
    body = {"error": message, "code": code}
    if details:
        body["details"] = details
    return jsonify(body), code


def register_error_handlers(app):
    @app.errorhandler(400)
    def bad_request(e):
        return _err("Bad Request", 400, str(e))

    @app.errorhandler(404)
    def not_found(e):
        return _err("Not Found", 404)

    @app.errorhandler(405)
    def method_not_allowed(e):
        return _err("Method Not Allowed", 405)

    @app.errorhandler(422)
    def unprocessable(e):
        return _err("Unprocessable Entity", 422, str(e))

    @app.errorhandler(500)
    def internal_error(e):
        return _err("Internal Server Error", 500)
