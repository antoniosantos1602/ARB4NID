import csv
import io

import pandas as pd
from models.data.datasetInteligence import DatasetIntelligence
from models.container import get_intelligence_cache


class AnalysisController:
    @staticmethod
    def analyze_csv(file):
        try:
            df = pd.read_csv(file)
            intelligence = DatasetIntelligence(df)
            get_intelligence_cache().save(intelligence)
            classes = sorted([str(c) for c in df['label'].unique().tolist()])
            all_cols = intelligence.feature_names
            bin_cols = intelligence.constraints.binary_columns
            continuous_cols = [
                c for c in all_cols
                if c not in bin_cols and df[c].nunique() > 1
            ]

            integer_count = len(intelligence.constraints.integer_columns)
            bin_count = len(intelligence.constraints.binary_columns)

            return {
                "status": "success",
                "classes": classes,
                "columns": intelligence.feature_names,
                "binary_columns": intelligence.constraints.binary_columns,
                "integer_columns": intelligence.constraints.integer_columns,
                "integer_count": integer_count,
                "bin_count" : bin_count,
                "continuous_cols":continuous_cols,
                "total_constraints": intelligence.constraints.total_dependencies,

                "unidirectional_dependencies": intelligence.constraints.unidirectional_dependencies,
                "mutual_exclusivity_dependencies": intelligence.constraints.mutual_exclusivity_dependencies,
                "linear_dependencies": intelligence.constraints.linear_dependencies
            }
        except Exception as e:
            print(f"Erro na análise: {str(e)}")
            raise Exception(f"Failed to analyze dataset: {str(e)}")

    @staticmethod
    def mahalanobis_groups_csv():
        intelligence = get_intelligence_cache().get()
        if not intelligence:
            raise Exception("No dataset analysed yet. Submit a dataset first.")

        mah = intelligence.mahalanobis_data
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["group_type", "label", "active_flags", "threshold"])

        for key, ref in mah.get("refs", {}).items():
            if not isinstance(ref, dict) or "threshold" not in ref:
                continue
            label = next((v for (k, v) in key if k == "label"), "")
            flags = "; ".join(f"{k}=1" for (k, v) in key if k != "label") or "-"
            writer.writerow(["label+flags", label, flags, round(float(ref["threshold"]), 5)])

        for label, ref in mah.get("label_refs", {}).items():
            writer.writerow(["label", label, "-", round(float(ref["threshold"]), 5)])

        gref = mah.get("global_ref")
        if gref:
            writer.writerow(["global", "-", "-", round(float(gref["threshold"]), 5)])

        return buf.getvalue()
