import numpy as np
import pandas as pd
from models.metrics.metrics import AdversarialContext
import matplotlib
matplotlib.use('Agg')
from models.data.feature_utils import get_integer_columns, get_binary_columns
import zipfile
import io


class AdversarialExporter:
    def __init__(self, attack_result):
        self.res = attack_result
        self.X_orig = attack_result.get("df_data_original")
        self.X_adv = np.asarray(attack_result.get("X_adv"))
        self.features_names = attack_result.get("column_names")

        self.context = AdversarialContext(
            y_true=attack_result.get("y_true"),
            y_pred_clean=attack_result.get("y_pred_clean"),
            targeted=attack_result.get("targeted", False),
            target_class=attack_result.get("target_class")
        )
        self.y_true = self.context.y_true
        self.y_pred_clean = self.context.y_pred_clean
        self.y_pred_adv = np.asarray(attack_result.get("y_pred_adv"))
        self.num_violations = np.asarray(attack_result.get("num_violacoes_all"))

        from utils.exporters.chart_renderer import ChartRenderer
        from utils.exporters.csv_exporter import CsvExporter
        self._charts = ChartRenderer(self.X_orig, self.X_adv, self.features_names)
        self._csv = CsvExporter(
            attack_result=attack_result,
            X_orig=self.X_orig,
            X_adv=self.X_adv,
            features_names=self.features_names,
            y_true=self.y_true,
            y_pred_clean=self.y_pred_clean,
            y_pred_adv=self.y_pred_adv,
            num_violations=self.num_violations,
        )

    def _get_n0_success_mask(self):
        universe = self.context.get_universe()
        mask_n0 = (self.num_violations == 0)
        if self.context.targeted:
            success = (self.y_pred_adv == self.context.target_class)
        else:
            success = (self.y_pred_adv != self.y_true)
        return universe & mask_n0 & success

    def _get_all_success_mask(self):
        universe = self.context.get_universe()
        if self.context.targeted:
            success = (self.y_pred_adv == self.context.target_class)
        else:
            success = (self.y_pred_adv != self.y_true)
        return universe & success

    def _format_confusion_matrix_md(self, cm):
        if cm is None:
            return "> Confusion matrix not available.\n"
        cm = np.array(cm, dtype=int)
        n = cm.shape[0]
        header = "| |" + "".join([f" Pred {i} |" for i in range(n)])
        sep    = "|---|" + "---|" * n
        rows   = [header, sep]
        for i, row in enumerate(cm):
            rows.append(f"| **Actual {i}** |" + "".join([f" {v} |" for v in row]))
        return "\n".join(rows) + "\n"

    def _plot_perturbation_subset(self, columns, title, output_path, color_pal="mako", mask=None):
        return self._charts.plot_perturbation_subset(columns, title, output_path, mask=mask, color_pal=color_pal)

    def _plot_radar_n0(self, columns, title, output_path):
        return self._charts.plot_radar_n0(columns, title, output_path, mask=self._get_n0_success_mask())

    def _plot_cbr_chart(self, cbr, output_path):
        return self._charts.plot_cbr_chart(cbr, output_path)

    def plot_mahalanobis(self, output_path):
        self._charts.plot_mahalanobis(self.res.get("mahalanobis_zones", {}), output_path)

    def _get_comparison_stats_for_mask(self, mask):
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return None

        X_orig_sub = self.X_orig.iloc[indices][self.features_names]
        X_adv_sub  = pd.DataFrame(self.X_adv[indices], columns=self.features_names)

        rows = []
        for col in self.features_names:
            orig = X_orig_sub[col]
            adv  = X_adv_sub[col]
            if np.abs(orig.values - adv.values).max() > 1e-6:
                rows.append({
                    "Feature":  col,
                    "Orig Max": f"{orig.max():.2f}",
                    "Adv Max":  f"{adv.max():.2f}",
                    "Orig Min": f"{orig.min():.2f}",
                    "Adv Min":  f"{adv.min():.2f}",
                    "Orig Avg": f"{orig.mean():.2f}",
                    "Adv Avg":  f"{adv.mean():.2f}",
                })
        return pd.DataFrame(rows)

    def _generate_conclusion(self, asr, adv_acc, ars, raa, cbr, dep_ausente):
        violated = []
        for ctype, label in [('binary','binary'), ('integer','integer'),
                              ('dependency','dependency'), ('mahalanobis','Mahalanobis')]:
            if cbr.get(ctype, {}).get('rate', 0) > 0.1:
                violated.append(label)

        violated_str = ", ".join(violated) if violated else "domain"

        gap = round(asr - ars, 2)
        realism_pct = round(raa, 2)

        if gap < 5:
            quality = "highly realistic — the generated perturbations respect domain constraints and represent genuine threats"
        elif gap < 20:
            quality = "moderately realistic — most perturbations are plausible but some violate domain constraints"
        else:
            quality = "largely unrealistic — a significant portion of the apparent successes violate domain constraints and would be detectable in practice"

        attack_name = self.res.get('attack_name', 'The attack')
        model_name  = self.res.get('model_name', 'the model')

        text = (
            f"Although **{attack_name}** fooled **{model_name}** in **{asr}%** of cases "
            f"(leaving the model with **{adv_acc}%** Adversarial Accuracy), "
            f"the attack is {quality}.\n\n"
            f"The realism assessment is computed via the **Attack Reputation Score (ARS)**, "
            f"which penalises adversarial successes that violate structural domain constraints — "
            f"specifically **{violated_str}** constraints in this run. "
            f"The gap between ASR and ARS is **{gap}%**, meaning that gap represents "
            f"attack successes that would not survive domain validation.\n\n"
            f"**ARS: {ars}%** · **Realistic Adversarial Accuracy (RAA): {raa}%**\n\n"
        )
        return text

    def generate_csv_all_success(self, output_path):
        self._csv.export_all_success(self._get_all_success_mask(), output_path)


    def generate_full_zip(self, zip_buffer):
        res = self.res
        attack_name = res.get('attack_name', 'Unknown')
        model_name  = res.get('model_name',  'Unknown')

        def _f(val, digits=2):
            try:
                return round(float(val), digits)
            except (TypeError, ValueError):
                return 0.0

        clean_acc  = _f(res.get('clean_accuracy', 0))
        exec_time  = _f(res.get('execution_time', 0))
        asr        = _f(res.get('asr', 0))
        adv_acc    = _f(res.get('adversarial_accuracy', 0))
        ars        = _f(float(res.get('wasr_v2', 0) or 0) * 100)
        raa        = _f(res.get('realistic_adversarial_accuracy', 0))

        nk          = res.get('nk', {})
        n_success   = int(res.get('n_success', 0) or 0)
        cbr         = res.get('constraint_breach_rate', {})
        pp          = res.get('penalty_percentiles', {})
        mah_zones   = res.get('mahalanobis_zones', {})
        dep_ausente = res.get('dep_ausente', True)
        n_bin_cols  = int(res.get('n_binary_cols', 0) or 0)
        n_int_cols  = int(res.get('n_integer_cols', 0) or 0)
        n0_count    = int(nk.get(0, nk.get('0', 0)))

        valid_set  = set(self.features_names)
        cols_bin   = [c for c in get_binary_columns(self.X_orig)  if c in valid_set]
        cols_int   = [c for c in get_integer_columns(self.X_orig) if c in valid_set]
        cols_cont  = [c for c in self.features_names if c not in cols_bin and c not in cols_int]

        all_success_mask = self._get_all_success_mask()

        lines = []

        lines += [f"# Analysis Report: {attack_name} — {model_name}\n"]

        lines += [
            "## Metadata\n",
            "| Field | Value |",
            "|---|---|",
            f"| **Model** | {model_name} |",
            f"| **Clean Accuracy** | {clean_acc}% |\n",
        ]

        lines += [
            f"## {attack_name}\n",
            "| Field | Value |",
            "|---|---|",
            f"| **Execution Time** | {exec_time}s |",
            f"| **Attack Success Rate (ASR)** | {asr}% |",
            f"| **Adversarial Accuracy (AA)** | {adv_acc}% |",
            f"| **Attack Reputation Score (ARS)** | {ars}% |",
            f"| **Realistic Adversarial Accuracy (RAA)** | {raa}% |\n",
        ]

        lines += ["### Adversarial Confusion Matrix\n"]
        lines += [self._format_confusion_matrix_md(res.get('adv_confusion_global'))]

        lines += ["## N₀–N₄ Distribution\n",
                  "| Level | Constraints Violated | Count | % of Successful |",
                  "|---|---|---|---|"]
        for k in range(5):
            cnt = int(nk.get(k, nk.get(str(k), 0)))
            pct = round(cnt / n_success * 100, 1) if n_success > 0 else 0.0
            lines.append(f"| **N{k}** | {k} | {cnt} | {pct}% |")
        lines.append("")

        lines += ["## Constraint Breach Rate\n",
                  "| Constraint Type | Breach Rate | Samples Affected |",
                  "|---|---|---|"]
        for ctype, label in [('binary','Binary'), ('integer','Integer'),
                              ('dependency','Dependency'), ('mahalanobis','Mahalanobis')]:
            d = cbr.get(ctype, {})
            if d:
                rate  = round(float(d.get('rate', 0)) * 100, 1)
                count = d.get('count', 0)
                lines.append(f"| **{label}** | {rate}% | {count} of {n_success} |")
        lines += ["", "![Constraint Breach Rate](images/cbr_chart.png)\n"]

        lines += ["## Binary Reputation Analysis\n"]
        bin_pp = pp.get('binary', {})
        if bin_pp and n_bin_cols > 0:
            lines += [f"Dataset has **{n_bin_cols}** binary features.\n",
                      "| Metric | Reputation (%) | Interpretation |",
                      "|---|---|---|"]
            for key, label in [('p25','P25'), ('p50','Median (P50)'), ('mean','Mean'), ('p75','P75')]:
                pen = _f(bin_pp.get(key, 0), 4)
                rep_pct = round((1 - pen) * 100, 1)
                lines.append(f"| **{label}** | {rep_pct}% | ~{round((1-pen) * n_bin_cols, 1)} of {n_bin_cols} binary features respected |")
        else:
            lines += ["> No binary constraint violations recorded.\n"]
        lines.append("")

        lines += ["## Integer Reputation Analysis\n"]
        int_pp = pp.get('integer', {})
        if int_pp and n_int_cols > 0:
            lines += [f"Dataset has **{n_int_cols}** integer features.\n",
                      "| Metric | Reputation (%) | Interpretation |",
                      "|---|---|---|"]
            for key, label in [('p25','P25'), ('p50','Median (P50)'), ('mean','Mean'), ('p75','P75')]:
                pen = _f(int_pp.get(key, 0), 4)
                rep_pct = round((1 - pen) * 100, 1)
                lines.append(f"| **{label}** | {rep_pct}% | ~{round((1-pen) * n_int_cols, 1)} of {n_int_cols} integer features respected |")
        else:
            lines += ["> No integer constraint violations recorded.\n"]
        lines.append("")

        lines += ["## Mahalanobis Outlier Analysis\n"]
        if mah_zones:
            mah_pp = pp.get('mahalanobis', {})
            lines += ["| Zone | Threshold | Count | Rate |",
                      "|---|---|---|---|"]
            for key, label, thresh in [('inside','Inside','r ≤ 1'),
                                        ('mild',  'Mild',  '1 < r ≤ 2'),
                                        ('severe','Severe','r > 2')]:
                d = mah_zones.get(key, {})
                lines.append(f"| **{label}** | {thresh} | {d.get('count',0)} | {round(float(d.get('rate',0))*100,1)}% |")
            if mah_pp:
                lines += ["", "**Mahalanobis Reputation:**",
                          "| Metric | Reputation (%) | Interpretation |",
                          "|---|---|---|"]
                for key, label in [('p25','P25'), ('p50','Median (P50)'), ('mean','Mean'), ('p75','P75')]:
                    pen = _f(mah_pp.get(key, 0), 4)
                    rep_pct = round((1 - pen) * 100, 1)
                    lines.append(f"| **{label}** | {rep_pct}% | {rep_pct}% of successful examples within the expected statistical distribution |")
            lines += ["", "![Mahalanobis Distribution](images/mahalanobis_dist.png)\n"]
        else:
            lines += ["> No Mahalanobis data available.\n"]

        lines += ["## Dependencies Reputation Analysis\n"]
        dep_subtypes = pp.get('dep_subtypes', {})
        dep_pp = pp.get('dependency', {})
        if not dep_ausente and dep_subtypes:
            if dep_pp:
                lines += ["| Metric | Reputation (%) | Interpretation |",
                          "|---|---|---|"]
                for key, label in [('p25','P25'), ('p50','Median (P50)'), ('mean','Mean'), ('p75','P75')]:
                    pen = _f(dep_pp.get(key, 0), 4)
                    rep_pct = round((1 - pen) * 100, 1)
                    lines.append(f"| **{label}** | {rep_pct}% | {rep_pct}% of dependency rules respected |")
                lines += [""]

            lines += ["| Sub-type | Total Rules | P25 violated | Median violated | Mean violated | P75 violated |",
                      "|---|---|---|---|---|---|"]
            for key, label in [('dep_uni','Unidirectional'),
                                ('dep_excl','Mutual Exclusivity'),
                                ('dep_lin','Linear')]:
                d = dep_subtypes.get(key, {})
                if d:
                    lines.append(
                        f"| **{label}** | {d.get('n_total',0)} | "
                        f"{_f(d.get('p25',0),2)} | {_f(d.get('p50',0),2)} | "
                        f"{_f(d.get('mean',0),2)} | {_f(d.get('p75',0),2)} |"
                    )
            lines.append("")

            validator = res.get("validator")
            if validator is not None:
                success_indices = np.where(all_success_mask)[0]
                detailed = validator.get_detailed_broken_rules(success_indices)

                uni_pairs  = set()
                excl_pairs = set()
                lin_pairs  = set()
                for rule_list in detailed.values():
                    for rule in rule_list:
                        if rule.startswith("Unidirectional"):
                            uni_pairs.add(rule)
                        elif rule.startswith("Mutual Exclusivity"):
                            excl_pairs.add(rule)
                        else:
                            lin_pairs.add(rule)

                if uni_pairs or excl_pairs or lin_pairs:
                    lines += ["### Broken Dependency Rules\n"]
                    for label, pairs in [
                        ("Unidirectional",     uni_pairs),
                        ("Mutual Exclusivity", excl_pairs),
                        ("Linear",             lin_pairs),
                    ]:
                        if pairs:
                            lines.append(f"**{label}:**")
                            for p in sorted(pairs):
                                lines.append(f"- {p}")
                            lines.append("")
        elif dep_ausente:
            lines += ["> No dependency constraints defined for this dataset.\n"]
        else:
            lines += ["> No dependency violations recorded.\n"]

        lines += ["## Perturbation Frequency\n",
                  "Frequency of feature modifications across all successful adversarial examples.\n",
                  "### Binary Features",
                  "![Binary Perturbation](images/pert_binary.png)\n",
                  "### Integer Features",
                  "![Integer Perturbation](images/pert_integer.png)\n",
                  "### Continuous Features",
                  "![Continuous Perturbation](images/pert_continuous.png)\n"]

        lines += ["## Descriptive Statistics: Original vs Adversarial\n"]
        all_mask = self._get_all_success_mask()
        df_stats = self._get_comparison_stats_for_mask(all_mask)
        if df_stats is not None and not df_stats.empty:
            lines += ["| Feature | Orig Max | Adv Max | Orig Min | Adv Min | Orig Avg | Adv Avg |",
                      "|---" * 7 + "|"]
            for _, row in df_stats.iterrows():
                lines.append(
                    f"| {row['Feature']} | {row['Orig Max']} | {row['Adv Max']} | "
                    f"{row['Orig Min']} | {row['Adv Min']} | {row['Orig Avg']} | {row['Adv Avg']} |"
                )
        else:
            lines += ["> No significant feature modifications detected.\n"]
        lines.append("")

        lines += ["## Most Perturbed Features for Realistic Examples (N₀)\n"]
        if n0_count > 0:
            lines += [
                f"Analysis based on **{n0_count}** successful examples with 0 constraint "
                f"violations (N₀ — fully realistic attacks).\n",
                "### Binary Features",
                "![N₀ Binary](images/n0_radar_binary.png)\n",
                "### Integer Features",
                "![N₀ Integer](images/n0_radar_integer.png)\n",
                "### Continuous Features",
                "![N₀ Continuous](images/n0_radar_continuous.png)\n",
            ]
        else:
            lines += [
                "> Não existem resultados disponíveis — o ataque não produziu exemplos "
                "adversariais sem violações de restrições de domínio.\n"
            ]

        lines += ["## Conclusão\n",
                  self._generate_conclusion(asr, adv_acc, ars, raa, cbr, dep_ausente)]

        full_content = "\n".join(lines)

        with zipfile.ZipFile(zip_buffer, "w") as zf:
            zf.writestr("REPORT.md", full_content)

            buf = io.BytesIO()
            if self._plot_cbr_chart(cbr, buf):
                zf.writestr("images/cbr_chart.png", buf.getvalue())

            buf = io.BytesIO()
            self.plot_mahalanobis(buf)
            if buf.getvalue():
                zf.writestr("images/mahalanobis_dist.png", buf.getvalue())

            for p_type, cols, pal in [
                ("binary",     cols_bin,  "Blues_r"),
                ("integer",    cols_int,  "Oranges_r"),
                ("continuous", cols_cont, "Greens_r"),
            ]:
                buf = io.BytesIO()
                if self._plot_perturbation_subset(
                    cols, f"Perturbation — {p_type.capitalize()} Features",
                    buf, color_pal=pal, mask=all_success_mask
                ):
                    zf.writestr(f"images/pert_{p_type}.png", buf.getvalue())

            if n0_count > 0:
                for p_type, cols in [
                    ("binary",     cols_bin),
                    ("integer",    cols_int),
                    ("continuous", cols_cont),
                ]:
                    if not cols:
                        continue
                    buf = io.BytesIO()
                    if self._plot_radar_n0(cols, f"N0 — {p_type.capitalize()} Feature Perturbation", buf):
                        zf.writestr(f"images/n0_radar_{p_type}.png", buf.getvalue())
