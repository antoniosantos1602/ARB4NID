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
