import functools
import threading

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


_RENDER_LOCK = threading.Lock()


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _RENDER_LOCK:
            return fn(*args, **kwargs)
    return wrapper


class ChartRenderer:
    def __init__(self, X_orig, X_adv, features_names) -> None:
        self.X_orig = X_orig
        self.X_adv = X_adv
        self.features_names = features_names

    @_locked
    def plot_perturbation_subset(self, columns, title, output_path, mask, color_pal: str = "mako") -> bool:
        if not columns:
            return False
        if mask is None:
            return False

        indices = np.where(mask)[0]
        n_sucesso = len(indices)
        if n_sucesso == 0:
            return False

        col_indices = [self.features_names.index(c) for c in columns]
        X_orig_k = self.X_orig.iloc[indices][columns].values
        X_adv_k = self.X_adv[indices][:, col_indices]
        diff = np.abs(X_adv_k - X_orig_k)
        perturbations = (diff > 0.1).sum(axis=0)
        freqs = (perturbations / n_sucesso) * 100

        df = pd.DataFrame({'Feature': columns, 'Freq': freqs})
        df = df[df['Freq'] > 0].sort_values(by='Freq', ascending=False).head(10)

        if df.empty:
            return False

        plt.figure(figsize=(10, 6))
        sns.set_style("whitegrid")
        ax = sns.barplot(data=df, x='Freq', y='Feature', palette=color_pal, hue='Feature', legend=False)
        plt.title(f"{title} (n={n_sucesso})", fontsize=12)
        plt.xlabel("Frequency of Modification (%)")
        plt.xlim(0, 115)
        for p in ax.patches:
            plt.gca().text(p.get_width() + 1, p.get_y() + p.get_height() / 2,
                           f'{p.get_width():.4f}%', va='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True

    @_locked
    def plot_radar_n0(self, columns, title, output_path, mask) -> bool:
        if not columns:
            return False
        if mask is None:
            return False

        indices = np.where(mask)[0]
        if len(indices) == 0:
            return False

        col_indices = [self.features_names.index(c) for c in columns]
        X_orig_k = self.X_orig.iloc[indices][columns].values
        X_adv_k = self.X_adv[indices][:, col_indices]
        diff = np.abs(X_adv_k - X_orig_k)
        freqs = ((diff > 0.1).sum(axis=0) / len(indices)) * 100

        df = pd.DataFrame({'Feature': columns, 'Freq': freqs})
        df = df[df['Freq'] > 0].sort_values(by='Freq', ascending=False).head(12)

        if df.empty:
            return False

        cats = df['Feature'].tolist()
        values = df['Freq'].tolist()
        N = len(cats)

        if N < 3:
            plt.figure(figsize=(8, 4))
            plt.barh(cats, values, color='#60a5fa')
            plt.xlabel("Modification Frequency (%)")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            return True

        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles_plot = angles + [angles[0]]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles_plot, values_plot, 'o-', linewidth=2, color='#60a5fa')
        ax.fill(angles_plot, values_plot, alpha=0.25, color='#60a5fa')
        ax.set_xticks(angles)
        ax.set_xticklabels(cats, size=9)
        ax.set_title(title, pad=20, fontsize=12)
        ax.set_ylim(0, 100)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True

    @staticmethod
    @_locked
    def plot_cbr_chart(cbr: dict, output_path) -> bool:
        labels = ['Binary', 'Integer', 'Dependency', 'Mahalanobis']
        keys = ['binary', 'integer', 'dependency', 'mahalanobis']
        values = [round(cbr.get(k, {}).get('rate', 0) * 100, 5) for k in keys]

        if all(v == 0 for v in values):
            return False

        colors = ['#60a5fa', '#fbbf24', '#c084fc', '#34d399']
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.barh(labels, values, color=colors)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Breach Rate (%)")
        ax.set_title("Constraint Breach Rate")
        for bar, val in zip(bars, values):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}%', va='center', fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return True

    @staticmethod
    @_locked
    def plot_mahalanobis(mahalanobis_zones: dict, output_path) -> None:
        if not mahalanobis_zones:
            return

        labels = ['Inside (r≤1)', 'Mild (1<r≤2)', 'Severe (r>2)']
        keys = ['inside', 'mild', 'severe']
        values = [round(mahalanobis_zones.get(k, {}).get('rate', 0) * 100, 5) for k in keys]
        colors = ['#334155', '#ca8a04', '#9f1239']

        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(labels, values, color=colors, edgecolor='black', alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1,
                    f'{val:.4f}%', ha='center', fontweight='bold')
        ax.set_ylabel("% of Successful Attacks")
        ax.set_title("Mahalanobis Zone Distribution")
        ax.set_ylim(0, 115)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
