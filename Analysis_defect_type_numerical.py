from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from typing import Mapping, Sequence,Tuple
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests

def type_defect_count_vs_factors(
    df,
    factors,
    defect_col='Defect_code_full',
    id_col='Pipe_ID',
    min_samples=15,
    method='spearman',
    material=None,
    material_col='Material',
    alpha=0.05,
    apply_mtc=True
):
    """
    Analyze the relationship between defect counts and pipe factors using correlation,
    applying FDR correction per material.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing defect and factor data.
        defect_col (str): Column name for defect types (default: 'Defect_code_full').
        id_col (str): Column name for unique pipe identifier (default: 'Pipe_ID').
        factors (list): List of column names for factors to analyze.
        min_samples (int): Minimum number of occurrences for a defect to be included.
        method (str): Correlation method, either 'spearman' or 'pearson'.
        material (str or list): Material(s) to filter by, or None for no filtering.
        material_col (str): Column name for material data.
        alpha (float): Significance threshold for FDR correction.

    Returns:
        rho_mat: DataFrame of correlation coefficients (defects x factors)
        pval_mat: DataFrame of raw p-values
        signif_mat: DataFrame of correlation strings with "*" for significant correlations after FDR
        reason_mat: DataFrame with notes on why a correlation might be missing
        total_occ: Series with total occurrences per defect
        pipe_factors: DataFrame with pipe-level factors
        counts_table: DataFrame with defect counts per pipe
    """

    # --- Filter by material ---
    if material is None:
        df_mat = df.copy()
        materials = df_mat[material_col].dropna().unique()
    elif isinstance(material, str):
        df_mat = df[df[material_col] == material].copy()
        materials = [material]
    else:
        df_mat = df[df[material_col].isin(material)].copy()
        materials = df_mat[material_col].dropna().unique()

    # --- Validate columns ---
    needed_cols = {defect_col, id_col, *factors, material_col}
    missing = [c for c in needed_cols if c not in df_mat.columns]
    if missing:
        raise ValueError(f"Missing columns in df: {missing}")

    # --- Prepare outputs ---
    rho_mat_total = pd.DataFrame(index=[], columns=factors, dtype=float)
    pval_mat_total = pd.DataFrame(index=[], columns=factors, dtype=float)
    signif_mat_total = pd.DataFrame(index=[], columns=factors, dtype=object)
    reason_mat_total = pd.DataFrame(index=[], columns=factors, dtype=object)
    total_occ_total = pd.Series(dtype=int)

    use_spearman = (method.lower() == 'spearman')

    for mat in materials:
        df_sub = df_mat[df_mat[material_col] == mat].copy()

        # Pipe-level factors
        pipe_factors = (df_sub[[id_col] + list(factors)]
                        .drop_duplicates(subset=[id_col])
                        .set_index(id_col)
                        .apply(lambda s: pd.to_numeric(s, errors='coerce')))

        # Counts per defect type per pipe
        counts_table = (df_sub.groupby([id_col, defect_col])
                             .size()
                             .unstack(fill_value=0))

        common_ids = pipe_factors.index.intersection(counts_table.index)
        pipe_factors = pipe_factors.loc[common_ids]
        counts_table = counts_table.loc[common_ids]

        # Only defects with sufficient total occurrences
        total_occ = counts_table.sum(axis=0)
        valid_defects = total_occ[total_occ >= min_samples].index.tolist()
        valid_defects = sorted(valid_defects, key=lambda x: total_occ[x], reverse=True)

        # Initialize matrices for this material
        rho_mat = pd.DataFrame(index=valid_defects, columns=factors, dtype=float)
        pval_mat = pd.DataFrame(index=valid_defects, columns=factors, dtype=float)
        reason_mat = pd.DataFrame(index=valid_defects, columns=factors, dtype=object)

        # --- Compute correlations ---
        for defect in valid_defects:
            y = counts_table[defect].astype(float)
            for f in factors:
                x = pipe_factors[f].astype(float)
                mask = x.notna() & y.notna()
                if mask.sum() <= 3:
                    rho, p, reason = np.nan, np.nan, f"Only {mask.sum()} valid pairs"
                elif x[mask].nunique() <= 1:
                    rho, p, reason = np.nan, np.nan, "X constant"
                elif y[mask].nunique() <= 1:
                    rho, p, reason = np.nan, np.nan, "Y constant"
                else:
                    if use_spearman:
                        rho, p = spearmanr(x[mask], y[mask])
                    else:
                        rho, p = pearsonr(x[mask], y[mask])
                    reason = "OK"
                rho_mat.loc[defect, f] = rho
                pval_mat.loc[defect, f] = p
                reason_mat.loc[defect, f] = reason

        # --- Apply multiple testing correction (optional) ---
        if apply_mtc:
            pvals = pval_mat.values.flatten()
            mask = ~np.isnan(pvals)

            pvals_adj = np.full_like(pvals, np.nan, dtype=float)

            if mask.sum() > 0:
                _, pvals_corr, _, _ = multipletests(
                    pvals[mask],
                    alpha=alpha,
                    method="fdr_bh"
                )
                pvals_adj[mask] = pvals_corr

            pval_corrected_mat = pd.DataFrame(
                pvals_adj.reshape(pval_mat.shape),
                index=pval_mat.index,
                columns=pval_mat.columns
            )
        else:
            pval_corrected_mat = pval_mat.copy()

        # --- Significance matrix ---
        signif_mat = rho_mat.copy().astype(str)

        for r in rho_mat.index:
            for c in rho_mat.columns:
                if pd.notna(rho_mat.loc[r, c]) and pval_corrected_mat.loc[r, c] < alpha:
                    signif_mat.loc[r, c] += "*"

        # --- Concatenate results across materials ---
        rho_mat_total = pd.concat([rho_mat_total, rho_mat])
        pval_mat_total = pd.concat([pval_mat_total, pval_corrected_mat])
        signif_mat_total = pd.concat([signif_mat_total, signif_mat])
        reason_mat_total = pd.concat([reason_mat_total, reason_mat])
        total_occ_total = pd.concat([total_occ_total, total_occ[valid_defects]])

    return rho_mat_total, pval_mat_total, signif_mat_total, reason_mat_total, total_occ_total, pipe_factors, counts_table

def plot_correlation_heatmaps_all_factors(
    df: pd.DataFrame,
    factors: Sequence[str],
    materials: Sequence[str] | None = None,
    method: str = "spearman",
    min_samples: int = 0,
    defect_col: str = "Defect_code_full",
    id_col: str = "Pipe_ID",
    material_col: str = "Material",
    p_threshold: float = 0.05,
    display_names: Mapping[str, str] | None = None,
):

    # ---------- 1) Defect order ----------
    defect_counts = df[defect_col].value_counts()
    all_valid_defects = defect_counts.index.tolist()

    # ---------- 2) Materials ----------
    if materials is not None:
        materials_all = list(materials)
    else:
        materials_all = sorted(
            df[material_col].dropna().astype(str).unique().tolist()
        )

    if not materials_all:
        raise ValueError("No materials found to plot.")

    # ---------- 3) Colormap ----------
    cmap = copy.copy(plt.get_cmap("coolwarm"))
    cmap.set_bad(color="lightgrey")
    vmin, vmax = -1, 1

    # ---------- 4) Compute correlations ----------
    rho_sig_per_mat = {}
    ever_significant = pd.Series(False, index=all_valid_defects, dtype=bool)

    for mat in materials_all:
        out = type_defect_count_vs_factors(
            df,
            defect_col=defect_col,
            id_col=id_col,
            factors=factors,
            min_samples=min_samples,
            method=method,
            material=mat,
            material_col=material_col,
        )

        rho_mat, pval_mat, *_ = out

        rho_mat = rho_mat.reindex(all_valid_defects)
        pval_mat = pval_mat.reindex(all_valid_defects)

        rho_sig = rho_mat.where(pval_mat < p_threshold)

        # track significance globally
        sig_mask = rho_sig.notna().any(axis=1)
        sig_mask = sig_mask.reindex(all_valid_defects, fill_value=False)
        ever_significant |= sig_mask

        rho_sig_per_mat[mat] = rho_sig

    # ---------- 5) Defects to use ----------
    defects_used = ever_significant[ever_significant].index.tolist()

    if not defects_used:
        print(
            "Warning: no defect is significant for any factor/material. "
            "Using all defects."
        )
        defects_used = all_valid_defects


    defects_used = [d for d in all_valid_defects if d in defects_used]

    # ---------- 6) Valid materials ----------
    valid_materials = [
        mat for mat in materials_all
        if rho_sig_per_mat[mat].loc[defects_used].notna().any().any()
    ]

    if not valid_materials:
        print("Warning: No significant correlations found for any material.")
        return

    # ---------- 7) Layout ----------
    ncols = 3
    nrows = int(np.ceil(len(valid_materials) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6 * ncols, 9 * nrows),
        sharey=False
    )

    axes = np.atleast_2d(axes)

    # ---------- 8) Plot ----------
    for idx, mat in enumerate(valid_materials):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]

        rho_sig = rho_sig_per_mat[mat].loc[defects_used]

        if display_names is not None:
            rho_sig = rho_sig.rename(columns=lambda x: display_names.get(x, x))

        sns.heatmap(
            rho_sig,
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=0,
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            ax=ax,
            linewidths=0.1,
            linecolor="white",
            annot_kws={"color": "black", "fontsize": 9},
        )

        ax.set_title(f"{mat}", fontsize=13, fontweight="bold")

        if c == 0:
            ax.set_yticklabels(rho_sig.index, rotation=0, fontsize=11)
        else:
            ax.set_yticks([])
            ax.set_ylabel("")

        ax.set_xticklabels(rho_sig.columns, rotation=45, ha="right", fontsize=11)

        ax.tick_params(
            axis="y",
            which="major",
            direction="out",
            length=4,
            width=0.8,
            left=True,
            right=False
        )


    # remove empty subplots
    total_slots = nrows * ncols
    for idx in range(len(valid_materials), total_slots):
        r = idx // ncols
        c = idx % ncols
        fig.delaxes(axes[r, c])

    # ---------- 9) Colorbar ----------
    cbar_ax = fig.add_axes([0.93, 0.25, 0.015, 0.5])

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax)

    cbar.set_label(
        "Spearman correlation coefficient",
        fontsize=13,
        labelpad=15
    )

    cbar.ax.tick_params(labelsize=11)

    fig.subplots_adjust(
        left=0.08,
        right=0.91,
        bottom=0.10,
        top=0.94,
        wspace=0.1,
        hspace=0.35
    )
    plt.show()
    return fig