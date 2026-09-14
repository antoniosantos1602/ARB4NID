import io
import tempfile

from flask import abort, send_file

from models.container import get_results_repository
from utils.export_utils import AdversarialExporter
from utils.exporters.model_report_exporter import generate_model_zip


class ExportController:

    @staticmethod
    def download_full_report(attack_id):
        res = get_results_repository().get(attack_id)
        if not res:
            abort(404, "Attack results not found")

        try:
            zip_buffer = io.BytesIO()
            exporter = AdversarialExporter(res)
            exporter.generate_full_zip(zip_buffer)

            zip_buffer.seek(0)

            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f"Full_Report_{attack_id}.zip"
            )
        except Exception as e:
            print(f"Error generating Full ZIP: {e}")
            abort(500, f"Error generating full report: {str(e)}")

    @staticmethod
    def download_csv_all(attack_id):
        res = get_results_repository().get(attack_id)
        if not res:
            abort(404, "Attack results not found")

        exporter = AdversarialExporter(res)
        tmp_path = tempfile.mktemp(suffix=".csv")
        exporter.generate_csv_all_success(tmp_path)

        attack_name = res.get('attack_name', attack_id).replace(' ', '_')
        return send_file(tmp_path, as_attachment=True,
                         download_name=f"adversarial_samples_{attack_name}.csv")

    @staticmethod
    def download_model_report(model_name):
        results = [r for r in get_results_repository().all() if r.get('model_name') == model_name]
        if not results:
            abort(404, f"No results found for model '{model_name}'")

        try:
            zip_buffer = io.BytesIO()
            generate_model_zip(results, zip_buffer)
            zip_buffer.seek(0)

            safe = model_name.replace(' ', '_').replace('/', '_')
            return send_file(
                zip_buffer,
                mimetype='application/zip',
                as_attachment=True,
                download_name=f"Model_Report_{safe}.zip"
            )
        except Exception as e:
            print(f"Error generating Model ZIP: {e}")
            abort(500, f"Error generating model report: {str(e)}")
