import io
import json

from flask import Blueprint, jsonify, request, send_file

from controllers.analysis_controller import AnalysisController
from controllers.attack_controller import AttackController
from controllers.export_controllers import ExportController
from models.attacks.catalog import ATTACKS_CATALOG
from utils.upload_handler import handle_upload

api_bp = Blueprint("api", __name__)


@api_bp.route("/api/attacks/available", methods=["GET"])
def get_available_attacks():
    return jsonify(ATTACKS_CATALOG)


@api_bp.route("/api/upload", methods=["POST"])
def upload_route():
    _, test_path, model_paths = handle_upload()
    return AttackController.run_attacks_stream(request.form, test_path, model_paths)


@api_bp.route("/api/results", methods=["GET"])
def list_results():
    return jsonify(AttackController.get_all_results())


@api_bp.route("/api/results/<attack_id>", methods=["DELETE"])
def delete_result_route(attack_id):
    result, status_code = AttackController.delete_result(attack_id)
    return jsonify(result), status_code


@api_bp.route("/api/results/clear-all", methods=["DELETE"])
def clear_all_results_route():
    result, status_code = AttackController.clear_all()
    return jsonify(result), status_code


@api_bp.route("/api/analyze-dataset", methods=["POST"])
def analyze_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    try:
        return jsonify(AnalysisController.analyze_csv(file))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/results/export-json", methods=["GET"])
def export_results():
    try:
        summary = AttackController.get_summary_export()
        if not summary:
            return jsonify({"error": "No results found"}), 404
        buf = io.BytesIO(json.dumps(summary, indent=4).encode("utf-8"))
        return send_file(buf, mimetype="application/json", as_attachment=True,
                         download_name="attack_summary_report.json")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@api_bp.route("/api/export/full-report/<attack_id>")
def download_full_report_route(attack_id):
    return ExportController.download_full_report(attack_id)


@api_bp.route("/api/export/csv/<attack_id>")
def download_csv_all_route(attack_id):
    return ExportController.download_csv_all(attack_id)


@api_bp.route("/api/export/model-report/<path:model_name>")
def download_model_report_route(model_name):
    return ExportController.download_model_report(model_name)
