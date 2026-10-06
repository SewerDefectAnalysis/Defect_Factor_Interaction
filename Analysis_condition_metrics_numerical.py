from typing import Mapping, Sequence,Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from scipy.stats import entropy, linregress, pearsonr, spearmanr, kruskal,chi2_contingency
from statsmodels.stats.multitest import multipletests


def plot_multimetric_correlation_heatmaps(
    df_defect,
    df_cctv_filtered,
    factors,
    display_names=None,
    materials: Sequence[str] = None,
    material_col: str = "Material",
    id_col: str = "Pipe_ID",
    cond_col: str = 'Condition_score',
    method: str = 'spearman',
    p_threshold: float = 0.05,
    show_only_significant: bool = True
):

    method = method.lower()
    if method not in ['spearman', 'pearson']:
        raise ValueError("method must be 'spearman' or 'pearson'")

    corr_func = spearmanr if method == 'spearman' else pearsonr

    if materials is None:
        materials = sorted(df_defect[material_col].dropna().unique())

    display_names = display_names or {f: f.replace('_', ' ') for f in factors}

    # -------------------------------------------------------------------------
    # Safe correlation extraction
    # -------------------------------------------------------------------------
    def safe_corr_result(result):
        if isinstance(result, (float, int, np.number)):
            return float(result), 1.0
        elif isinstance(result, tuple) and len(result) == 2:
            rho, p = result
            return float(rho), float(p)
        if hasattr(result, 'correlation') and hasattr(result, 'pvalue'):
            return float(result.correlation), float(result.pvalue)
        return np.nan, np.nan

    # -------------------------------------------------------------------------
    # Compute correlation by material WITH FDR per material
    # -------------------------------------------------------------------------
    def compute_correlation_by_material(df, y_col, factors, materials, material_col, p_threshold):

        rho_mat = pd.DataFrame(index=factors, columns=materials, dtype=float)
        pval_raw = pd.DataFrame(index=factors, columns=materials, dtype=float)
        pval_adj = pd.DataFrame(index=factors, columns=materials, dtype=float)

        for mat in materials:

            df_mat = df[df[material_col] == mat].copy()

            if len(df_mat) < 10:
                continue

            for f in factors:

                if f not in df_mat.columns:
                    continue

                x = pd.to_numeric(df_mat[f], errors='coerce')
                y = pd.to_numeric(df_mat[y_col], errors='coerce')

                mask = x.notna() & y.notna()

                if mask.sum() < 5:
                    continue

                result = corr_func(x[mask], y[mask])
                rho, p = safe_corr_result(result)

                rho_mat.loc[f, mat] = rho
                pval_raw.loc[f, mat] = p

            # FDR correction per material
            valid = pval_raw[mat].dropna()

            if len(valid) > 0:
                _, corrected, _, _ = multipletests(
                    valid.values,
                    alpha=p_threshold,
                    method='fdr_bh'
                )
                pval_adj.loc[valid.index, mat] = corrected

        rho_sig = rho_mat.copy()
        rho_sig[pval_adj >= p_threshold] = np.nan

        return rho_sig, rho_mat, pval_raw, pval_adj

    # -------------------------------------------------------------------------
    # 1. Prepare metrics
    # -------------------------------------------------------------------------
    defects_per_pipe = (
        df_defect.groupby([id_col, material_col]).size()
        .reset_index(name='DEFECTS_PER_PIPE')
    )

    factors_df = df_defect[[id_col] + factors + [material_col]].drop_duplicates(subset=id_col)

    df_pipe = defects_per_pipe.merge(factors_df, on=[id_col, material_col], how='left')

    df_cond = df_cctv_filtered.copy()

    # -------------------------------------------------------------------------
    # 2. Compute correlations per metric
    # -------------------------------------------------------------------------
    results = {}

    metrics = [
        ('Defects_per_pipe', df_pipe, 'DEFECTS_PER_PIPE'),
        ('Condition score',  df_cond,  cond_col)
    ]

    ever_significant = pd.Series(False, index=factors, dtype=bool)

    for key, df_in, col_y in metrics:

        rho_sig, rho_mat, pval_raw, pval_adj = compute_correlation_by_material(
            df_in,
            y_col=col_y,
            factors=factors,
            materials=materials,
            material_col=material_col,
            p_threshold=p_threshold
        )

        results[key] = {
            'rho': rho_mat,
            'pval_raw': pval_raw,
            'pval_adj': pval_adj,
            'rho_sig': rho_sig
        }

        sig_this_metric = rho_sig.notna().any(axis=1)
        sig_this_metric = sig_this_metric.reindex(factors, fill_value=False)
        ever_significant |= sig_this_metric

    if show_only_significant:
        factors_used = ever_significant[ever_significant].index.tolist()
        factors_used = [f for f in factors if f in factors_used]
    else:
        factors_used = factors.copy()

    # -------------------------------------------------------------------------
    # 3. Plot heatmaps
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True)

    vmin, vmax = -1, 1
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="lightgrey")

    metric_titles = [
        "Defects per pipe",
        "Condition score"
    ]

    y_labels = [display_names.get(f, f) for f in factors_used]

    # Add a dash after each factor label
    y_labels_with_dash = [f"{label} -" for label in y_labels]

    for ax, (key, title) in zip(
        axes,
        zip(results.keys(), metric_titles)
    ):
        rho_sig = results[key]["rho_sig"].loc[factors_used]

        sns.heatmap(
            rho_sig,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=0,
            linewidths=0.2,
            linecolor="white",
            cbar=False,
            ax=ax,
            annot_kws={
                "color": "black",
                "fontsize": 12
            }
        )

        # X-axis labels
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=0,
            ha="center",
            fontsize=13
        )

        # Y-axis labels
        ax.set_yticks(np.arange(len(factors_used)) + 0.5)
        ax.set_yticklabels(
            y_labels_with_dash,
            rotation=0,
            fontsize=14,
            va="center"
        )

        # Remove actual tick marks because the dash is included in the text
        ax.tick_params(
            axis="y",
            left=False,
            right=False,
            pad=0
        )

        ax.set_title(
            title,
            fontsize=15,
            pad=10,
            fontweight="bold"
        )

        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.spines["bottom"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)

        ax.tick_params(
            axis="x",
            which="major",
            bottom=True,
            top=False,
            direction="out",
            length=3,
            width=0.8,
            pad=2
        )

    # -------------------------------------------------------------------------
    # Colorbar with label
    # -------------------------------------------------------------------------
    cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.76])

    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=norm
    )
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        cax=cbar_ax
    )

    cbar.ax.tick_params(labelsize=12)

    if method == "spearman":
        cbar.set_label(
            "Spearman correlation coefficient",
            fontsize=14.5,
            labelpad=10
        )
    else:
        cbar.set_label(
            "Pearson correlation coefficient",
            fontsize=14.5,
            labelpad=10
        )

    fig.subplots_adjust(
        left=0.08,
        right=0.90,
        bottom=0.08,
        top=0.95,
        wspace=0.05,
        hspace=0.25
    )
    plt.rcParams["figure.dpi"] = 200
    plt.show()


    return fig, results
