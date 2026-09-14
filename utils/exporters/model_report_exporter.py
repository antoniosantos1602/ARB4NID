import io
import zipfile

import numpy as np

from models.data.feature_utils import get_binary_columns, get_integer_columns
from utils.export_utils import AdversarialExporter


def generate_model_zip(results_list, zip_buffer):
    if not results_list:
        return

    model_name = results_list[0].get('model_name', 'Unknown')

    def _f(val, digits=5):
        try:
            return round(float(val), digits)
        except (TypeError, ValueError):
            return 0.0

    lines = [f"# Model Robustness Report: {model_name}\n"]

    lines += ["## Executive Summary\n",
              "| Attack | Clean Acc | Exec Time | ASR | Adv. Accuracy | ARS | RAA |",
              "|---|---|---|---|---|---|---|"]
    for res in results_list:
        atk       = res.get('attack_name', '?')
        clean_acc = _f(res.get('clean_accuracy', 0))
        exec_time = _f(res.get('execution_time', 0))
        asr       = _f(res.get('asr', 0))
        adv_acc   = _f(res.get('adversarial_accuracy', 0))
        ars       = _f(float(res.get('wasr_v2', 0) or 0) * 100)
        raa       = _f(res.get('realistic_adversarial_accuracy', 0))
        lines.append(f"| **{atk}** | {clean_acc}% | {exec_time}s | {asr}% | {adv_acc}% | {ars}% | {raa}% |")
    lines.append("")

    all_images = {}

    for res in results_list:
        attack_name = res.get('attack_name', 'Unknown')
        safe_name   = attack_name.replace(' ', '_').replace('/', '_')
        img_dir     = f"images/{safe_name}"

        exporter = AdversarialExporter(res)

        clean_acc  = _f(res.get('clean_accuracy', 0))
        exec_time  = _f(res.get('execution_time', 0))
        asr        = _f(res.get('asr', 0))
        adv_acc    = _f(res.get('adversarial_accuracy', 0))
        ars        = _f(float(res.get('wasr_v2', 0) or 0) * 100)
        raa        = _f(res.get('realistic_adversarial_accuracy', 0))
        nk         = res.get('nk', {})
        n_success  = int(res.get('n_success', 0) or 0)
        cbr        = res.get('constraint_breach_rate', {})
        pp         = res.get('penalty_percentiles', {})
        mah_zones  = res.get('mahalanobis_zones', {})
        dep_ausente= res.get('dep_ausente', True)
        n_bin_cols = int(res.get('n_binary_cols', 0) or 0)
        n_int_cols = int(res.get('n_integer_cols', 0) or 0)
        n0_count   = int(nk.get(0, nk.get('0', 0)))

        valid_set    = set(exporter.features_names)
        cols_bin     = [c for c in get_binary_columns(exporter.X_orig)  if c in valid_set]
        cols_int     = [c for c in get_integer_columns(exporter.X_orig) if c in valid_set]
        cols_cont    = [c for c in exporter.features_names if c not in cols_bin and c not in cols_int]
        all_success_mask = exporter._get_all_success_mask()

        lines += [
            "---\n",
            f"## Attack: {attack_name}\n",
            "| Field | Value |",
            "|---|---|",
            f"| **Execution Time** | {exec_time}s |",
            f"| **Attack Success Rate (ASR)** | {asr}% |",
            f"| **Adversarial Accuracy (AA)** | {adv_acc}% |",
            f"| **Attack Reputation Score (ARS)** | {ars}% |",
            f"| **Realistic Adversarial Accuracy (RAA)** | {raa}% |\n",
        ]

        lines += ["### Adversarial Confusion Matrix\n"]
        lines += [exporter._format_confusion_matrix_md(res.get('adv_confusion_global'))]

        lines += ["### N₀–N₄ Distribution\n",
                  "| Level | Constraints Violated | Count | % of Successful |",
                  "|---|---|---|---|"]
        for k in range(5):
            cnt = int(nk.get(k, nk.get(str(k), 0)))
            pct = round(cnt / n_success * 100, 5) if n_success > 0 else 0.0
            lines.append(f"| **N{k}** | {k} | {cnt} | {pct}% |")
        lines.append("")

        lines += ["### Constraint Breach Rate\n",
                  "| Constraint Type | Breach Rate | Samples Affected |",
                  "|---|---|---|"]
        for ctype, label in [('binary','Binary'), ('integer','Integer'),
                              ('dependency','Dependency'), ('mahalanobis','Mahalanobis')]:
            d = cbr.get(ctype, {})
            if d:
                rate  = round(float(d.get('rate', 0)) * 100, 5)
                count = d.get('count', 0)
                lines.append(f"| **{label}** | {rate}% | {count} of {n_success} |")
        lines += ["", f"![Constraint Breach Rate]({img_dir}/cbr_chart.png)\n"]

        lines += ["### Binary Reputation Analysis\n"]
        bin_pp = pp.get('binary', {})
        if bin_pp and n_bin_cols > 0:
            lines += [f"Dataset has **{n_bin_cols}** binary features.\n",
                      "| Metric | Reputation | Interpretation |",
                      "|---|---|---|"]
            for key, label in [('p25','P25'), ('p50','Median (P50)'), ('mean','Mean'), ('p75','P75')]:
                pen = _f(bin_pp.get(key, 0), 4)
                rep_pct = round((1 - pen), 4)
                lines.append(f"| **{label}** | {rep_pct} | ~{round((1-pen)*n_bin_cols,2)} of {n_bin_cols} binary features respected |")
        else:
            lines += ["> No binary constraint violations recorded.\n"]
        lines.append("")

        lines += ["### Integer Reputation Analysis\n"]
        int_pp = pp.get('integer', {})
        if int_pp and n_int_cols > 0:
            lines += [f"Dataset has **{n_int_cols}** integer features.\n",
                      "| Metric | Reputation | Interpretation |",
                      "|---|---|---|"]
            for key, label in [('p25','P25'), ('p50','Median (P50)'), ('mean','Mean'), ('p75','P75')]:
                pen = _f(int_pp.get(key, 0), 4)
                rep_pct = round((1 - pen), 4)
                lines.append(f"| **{label}** | {rep_pct} | ~{round((1-pen)*n_int_cols,2)} of {n_int_cols} integer features respected |")
        else:
            lines += ["> No integer constraint violations recorded.\n"]
        lines.append("")

        lines += ["### Mahalanobis Outlier Analysis\n"]
        if mah_zones:
            mah_pp = pp.get('mahalanobis', {})
            lines += ["| Zone | Threshold | Count | Rate |",
                      "|---|---|---|---|"]
            for key, label, thresh in [('inside','Inside','r ≤ 1'),
                                        ('mild',  'Mild',  '1 < r ≤ 2'),
                                        ('severe','Severe','r > 2')]:
                d = mah_zones.get(key, {})
                lines.append(f"| **{label}** | {thresh} | {d.get('count',0)} | {round(float(d.get('rate',0))*100,5)}% |")
            if mah_pp:
                lines += ["", "**Mahalanobis Reputation:**",
                          "| Metric | Reputation | Interpretation |",
                          "|---|---|---|"]
                for key, label in [('p25','P25'), ('p50','Median (P50)'), ('mean','Mean'), ('p75','P75')]:
                    pen = _f(mah_pp.get(key, 0), 4)
                    rep_pct = round((1 - pen), 4)
                    lines.append(f"| **{label}** | {rep_pct} | {rep_pct} of successful examples within the expected statistical distribution |")
            lines += ["", f"![Mahalanobis Distribution]({img_dir}/mahalanobis_dist.png)\n"]
        else:
            lines += ["> No Mahalanobis data available.\n"]

        lines += ["### Dependencies Reputation Analysis\n"]
        dep_subtypes = pp.get('dep_subtypes', {})
        dep_pp = pp.get('dependency', {})
        if not dep_ausente and dep_subtypes:
            if dep_pp:
                lines += ["| Metric | Reputation | Interpretation |",
                          "|---|---|---|"]
                for key, label in [('p25','P25'), ('p50','Median (P50)'), ('mean','Mean'), ('p75','P75')]:
                    pen = _f(dep_pp.get(key, 0), 4)
                    rep_pct = round((1 - pen), 4)
                    lines.append(f"| **{label}** | {rep_pct} | {rep_pct} of dependency rules respected |")
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
                    lines += ["#### Broken Dependency Rules\n"]
                    for label, pairs in [("Unidirectional", uni_pairs),
                                         ("Mutual Exclusivity", excl_pairs),
                                         ("Linear", lin_pairs)]:
                        if pairs:
                            lines.append(f"**{label}:**")
                            for p in sorted(pairs):
                                lines.append(f"- {p}")
                            lines.append("")
        elif dep_ausente:
            lines += ["> No dependency constraints defined for this dataset.\n"]
        else:
            lines += ["> No dependency violations recorded.\n"]

        lines += ["### Perturbation Frequency\n",
                  "Frequency of feature modifications across all successful adversarial examples.\n",
                  "#### Binary Features",
                  f"![Binary Perturbation]({img_dir}/pert_binary.png)\n",
                  "#### Integer Features",
                  f"![Integer Perturbation]({img_dir}/pert_integer.png)\n",
                  "#### Continuous Features",
                  f"![Continuous Perturbation]({img_dir}/pert_continuous.png)\n"]

        lines += ["### Descriptive Statistics: Original vs Adversarial\n"]
        df_stats = exporter._get_comparison_stats_for_mask(all_success_mask)
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

        lines += ["### Most Perturbed Features for Realistic Examples (N₀)\n"]
        if n0_count > 0:
            lines += [
                f"Analysis based on **{n0_count}** successful examples with 0 constraint violations.\n",
                "#### Binary Features",
                f"![N₀ Binary]({img_dir}/n0_radar_binary.png)\n",
                "#### Integer Features",
                f"![N₀ Integer]({img_dir}/n0_radar_integer.png)\n",
                "#### Continuous Features",
                f"![N₀ Continuous]({img_dir}/n0_radar_continuous.png)\n",
            ]
        else:
            lines += ["> No adversarial examples without domain constraint violations were produced by this attack.\n"]

        lines += ["### Conclusion\n",
                  exporter._generate_conclusion(asr, adv_acc, ars, raa, cbr, dep_ausente),
                  ""]

        buf = io.BytesIO()
        if exporter._plot_cbr_chart(cbr, buf):
            all_images[f"{img_dir}/cbr_chart.png"] = buf.getvalue()

        buf = io.BytesIO()
        exporter.plot_mahalanobis(buf)
        if buf.getvalue():
            all_images[f"{img_dir}/mahalanobis_dist.png"] = buf.getvalue()

        for p_type, cols, pal in [
            ("binary",     cols_bin,  "Blues_r"),
            ("integer",    cols_int,  "Oranges_r"),
            ("continuous", cols_cont, "Greens_r"),
        ]:
            buf = io.BytesIO()
            if exporter._plot_perturbation_subset(
                cols, f"Perturbation — {p_type.capitalize()} Features",
                buf, color_pal=pal, mask=all_success_mask
            ):
                all_images[f"{img_dir}/pert_{p_type}.png"] = buf.getvalue()

        if n0_count > 0:
            for p_type, cols in [
                ("binary",     cols_bin),
                ("integer",    cols_int),
                ("continuous", cols_cont),
            ]:
                if not cols:
                    continue
                buf = io.BytesIO()
                if exporter._plot_radar_n0(cols, f"N0 — {p_type.capitalize()} Feature Perturbation", buf):
                    all_images[f"{img_dir}/n0_radar_{p_type}.png"] = buf.getvalue()

    with zipfile.ZipFile(zip_buffer, "w") as zf:
        zf.writestr("REPORT.md", "\n".join(lines))
        for path, data in all_images.items():
            zf.writestr(path, data)
