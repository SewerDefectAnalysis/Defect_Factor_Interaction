from scipy.stats import linregress
from typing import Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from typing import Mapping, Sequence, Tuple
import textwrap
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
from Config import DEFAULT_MIN_OBSERVATIONS_PER_GROUP, DISPLAY_NAMES
from Group_factors import group_cols
import re
from Config import build_material_color_map
from matplotlib.lines import Line2D


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
        min_samples: int = DEFAULT_MIN_OBSERVATIONS_PER_GROUP,
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

    # ---------- 5) Defects to use----------
    defects_used = ever_significant[ever_significant].index.tolist()

    if not defects_used:
        print(
            "Warning: no defect is significant for any factor/material. "
            "Using all defects."
        )
        defects_used = all_valid_defects

    # keep consistent order
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
    ncols = 4
    nrows = int(np.ceil(len(valid_materials) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(5 * ncols, 8 * nrows),
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
        ax.set_title(
            title,
            fontsize=14,
            pad=10
        )

        # ax.tick_params(
        #     axis="x",
        #     which="major",
        #     bottom=True,
        #     top=False,
        #     direction="out",
        #     length=5,
        #     width=1
        # )

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

# ----------------- Helper Functions -----------------

def build_corr_long_from_mats(rho_mat, pval_mat, material=None):
    """
    Convert correlation and p-value matrices into a long-format DataFrame.

    Parameters:
        rho_mat (pd.DataFrame): Matrix of correlation coefficients with defects as rows and factors as columns.
        pval_mat (pd.DataFrame): Matrix of p-values corresponding to the correlations in rho_mat.
        material (str, optional): Material name to add as a column in the output (default: None).

    Returns:
        corr_df (pd.DataFrame): Long-format DataFrame with columns 'Defect', 'Factor', 'rho', 'p', and optionally 'MATERIAL'.
    """

    # --- Convert correlation matrix to long format ---
    # Melt rho_mat to transform from wide to long format, renaming the index to 'Defect' and columns to 'Factor'
    rho_long = rho_mat.reset_index().melt(id_vars='index', var_name='Factor', value_name='rho') \
                      .rename(columns={'index':'Defect'})

    # --- Convert p-value matrix to long format ---
    # Melt pval_mat similarly to create a long-format DataFrame for p-values
    p_long = pval_mat.reset_index().melt(id_vars='index', var_name='Factor', value_name='p') \
                      .rename(columns={'index':'Defect'})

    # --- Merge correlation and p-value DataFrames ---
    # Combine the two long-format DataFrames on 'Defect' and 'Factor' columns
    corr_df = rho_long.merge(p_long, on=['Defect','Factor'])

    # --- Add material column if specified ---
    # Include a 'MATERIAL' column with the provided material name, if any
    if material is not None:
        corr_df['MATERIAL'] = material

    # --- Return the combined DataFrame ---
    return corr_df

def _mid_from_label(g):
    """
    Converts a group label (Interval, 'a-b', or 'x+') to its numerical midpoint.

    Parameters:
        g: A group label, either a pandas Interval or a string like '10-20' or '50+'.
    Returns:
        float: The midpoint value of the interval or number, or np.nan if invalid.
    """
    # Midpoint of a pandas Interval
    if isinstance(g, pd.Interval):
        return (g.left + g.right) / 2.0
    s = str(g).replace('–', '-').replace('—', '-').strip()
    m_plus = re.match(r'^\s*(\d+(?:\.\d+)?)\s*\+$', s)
    if m_plus:
        return float(m_plus.group(1))
    nums = re.findall(r'(\d+(?:\.\d+)?)', s)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2.0
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


def _resolve_group_col(factor, df, group_cols):
    """
    Finds the corresponding *_GROUP column for a given factor.

    Parameters:
        factor: The factor name (e.g., 'AGE', 'LENGTH').
        df: DataFrame containing the data.
        group_cols: Dictionary mapping factors to group columns or a list of factors.
    Returns:
        str or None: The group column name if found, else None.
    """
    # Explicit mapping
    if isinstance(group_cols, dict):
        cand = group_cols.get(factor)
        if cand in df.columns:
            return cand
    # Convention: <factor>_GROUP
    auto = f"{factor}_GROUP"
    if auto in df.columns:
        return auto
    f = factor.lower()
    if 'year' in f and 'AGE_GROUP' in df.columns:
        return 'AGE_GROUP'
    if 'age' in f:
        for c in df.columns:
            if c.lower().endswith('_group') and 'year' in c.lower():
                return c
    # Loose matching
    fkey = f.replace('_', '')
    matches = [c for c in df.columns if c.endswith('_GROUP') and fkey in c.lower().replace('_', '')]
    return matches[0] if matches else None


def _points_for_plot_grouped(df, defect, group_col,
                             id_col='COMPKEY', defect_col='DEFE_MAIN',
                             material=None, material_col='MATERIAL',
                             min_pipes=3,
                             low_n_threshold=15):
    """
    Build one point per group: x_plot, mean_defects, n_pipes, and a low_n flag.

    Parameters
    ----------
    df : pd.DataFrame
        Source table with pipe rows and (possibly repeated) defect rows.
    defect : str
        Specific defect code to count per pipe (e.g., 'JF', 'DG', etc.).
    group_col : str
        Grouping column name (e.g., 'AGE_GROUP').
    id_col : str, default 'COMPKEY'
        Pipe identifier column.
    defect_col : str, default 'DEFE_MAIN'
        Column containing the defect type code per observation/row.
    material : str or None, default None
        If provided, filter to a single material before grouping.
    material_col : str, default 'MATERIAL'
        Column containing the material.
    min_pipes : int, default 3
        Absolute minimum number of pipes required for a group to exist as a point
        (prevents showing very spurious groups).
    low_n_threshold : int, default 15
        Threshold below which a group is considered "low sample size" (low_n=True).

    Returns
    -------
    pd.DataFrame
        Columns ['GROUP', 'x_plot', 'mean_defects', 'n_pipes', 'low_n'].
    """
    sub = df if material is None else df[df[material_col] == material]
    if group_col not in sub.columns:
        return pd.DataFrame(columns=['GROUP', 'x_plot', 'mean_defects', 'n_pipes', 'low_n'])

    # Count target defect occurrences per pipe (0 if not present)
    counts = (sub[sub[defect_col] == defect]
              .groupby(id_col).size()
              .reindex(sub[id_col].drop_duplicates(), fill_value=0)
              .rename('count_defect'))

    # One row per pipe with its group, then join counts
    base = (sub[[id_col, group_col]]
            .drop_duplicates(subset=[id_col])
            .set_index(id_col))
    merged = base.join(counts, how='left').reset_index()

    # Aggregate per group: mean defects per pipe and number of pipes
    agg = (merged.groupby(group_col, dropna=False, observed=False)
                 .agg(mean_defects=('count_defect', 'mean'),
                      n_pipes=(id_col, 'nunique'))
                 .reset_index()
                 .rename(columns={group_col: 'GROUP'}))

    # Keep only groups with a minimal absolute size (avoid 1-pipe noise, etc.)
    agg = agg[agg['n_pipes'] >= min_pipes].copy()

    # Group midpoint for x coordinate (expects labels like '0-20', '20-40', etc.)
    agg['x_plot'] = agg['GROUP'].apply(_mid_from_label)
    agg = agg.dropna(subset=['x_plot']).sort_values('x_plot').reset_index(drop=True)

    # Flag for low sample size (to style in grey and/or exclude from fit)
    agg['low_n'] = agg['n_pipes'] < low_n_threshold

    return agg[['GROUP', 'x_plot', 'mean_defects', 'n_pipes', 'low_n']]


def plot_top_type_defects(
    df,
    corr_df,
    group_cols=group_cols,
    materials=None,
    id_col='Pipe_ID',
    defect_col='Defect_code',
    material_col='Material',
    top_k=3,
    p_thr=0.05,
    min_pipes_per_group=3,
    low_n_threshold=15,
    min_groups_to_show=2,
    min_groups_for_fit=3,
    factor_renamer=DISPLAY_NAMES,
    cell_size=(4.5, 3.5),
    colors_materials=None,
    color_by_material=True,
    point_style="Grey",
    fit_mode="exclude",
    show_point_labels=False
):
    """
    Plot top-k defect-factor relationships for each material.

    Each panel shows the mean number of defects per pipe for grouped values
    of a numerical factor. A linear regression is fitted when enough valid
    groups are available.

    Parameters
    ----------
    df : pd.DataFrame
        Main dataset.

    corr_df : pd.DataFrame
        Correlation results with columns:
        ['MATERIAL', 'Defect', 'Factor', 'rho', 'p'].

    group_cols : dict | list | None
        Dictionary mapping each factor to its grouped column.
        If a list is provided, '<factor>_GROUP' is assumed.

    materials : list | None
        Materials to plot. If None, materials are inferred from corr_df.

    id_col : str
        Pipe identifier column.

    defect_col : str
        Defect type column.

    material_col : str
        Material column.

    top_k : int
        Maximum number of defect-factor relationships shown per material.

    p_thr : float
        Significance threshold used to select correlations.

    min_pipes_per_group : int
        Minimum number of pipes required for a group to be plotted.

    low_n_threshold : int
        Groups with fewer pipes than this threshold are considered low-sample
        groups and can be displayed in grey.

    min_groups_to_show : int
        Minimum number of groups required to display a panel.

    min_groups_for_fit : int
        Minimum number of valid groups required to fit a regression.

    factor_renamer : dict | None
        Dictionary used to rename factors for display.

    cell_size : tuple
        Width and height of each subplot.

    colors_materials : dict | None
        Dictionary mapping materials to colors.

        Example:
        {
            'AC': '#1f77b4',
            'PVC': '#ff7f0e',
            'CONC': '#2ca02c'
        }

    color_by_material : bool
        If True, use colors_materials to color each material.
        Otherwise, use steelblue.

    point_style : str
        If "Grey", low-sample groups are displayed in grey.

    fit_mode : str
        "exclude": low-sample groups are excluded from regression.
        "include": low-sample groups are included in regression.

    show_point_labels : bool
        If True, display mean defect values above the points.

    Returns
    -------
    pd.DataFrame
        Regression results for panels where a regression could be fitted.
    """

    # Normalize group_cols to a dictionary
    if group_cols is None:
        group_cols = {}

    elif isinstance(group_cols, (list, tuple, set)):
        group_cols = {
            factor: f"{factor}_GROUP"
            for factor in group_cols
        }

    elif not isinstance(group_cols, dict):
        raise TypeError(
            "group_cols must be a dict, list, tuple, set, or None."
        )

    # Validate fit_mode
    if fit_mode.lower() not in {"include", "exclude"}:
        raise ValueError(
            "fit_mode must be either 'include' or 'exclude'."
        )

    # Resolve the grouping column associated with each factor
    def _resolve_group_col(factor, df_local, mapping):

        candidate = mapping.get(factor)

        if candidate and candidate in df_local.columns:
            return candidate

        alternative = f"{factor}_GROUP"

        if alternative in df_local.columns:
            return alternative

        return None

    # Determine materials to plot
    mats = (
        materials
        if materials is not None
        else list(corr_df['MATERIAL'].dropna().unique())
    )

    if len(mats) == 0:
        raise ValueError("No materials are available to plot.")

    n_cols = len(mats)
    n_rows = top_k

    # Validate material colors
    if color_by_material:

        if colors_materials is None:
            raise ValueError(
                "colors_materials must be provided when "
                "color_by_material=True."
            )

        missing_colors = [
            mat
            for mat in mats
            if mat not in colors_materials
        ]

        if missing_colors:
            raise ValueError(
                "Missing colors in colors_materials for: "
                + ", ".join(map(str, missing_colors))
            )

    # Create subplot grid
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(
            n_cols * cell_size[0],
            n_rows * cell_size[1]
        ),
        squeeze=False
    )

    results = []

    use_grey = (
        str(point_style).lower() == "grey"
    )

    # Store legend information for a single figure-level legend
    legend_handles = None
    legend_labels = None

    for j, mat in enumerate(mats):

        # Select significant relationships for the current material
        sub = corr_df[
            (corr_df['MATERIAL'] == mat)
            & (corr_df['p'] < p_thr)
        ].copy()

        if sub.empty:

            for r in range(n_rows):
                axes[r, j].axis('off')

            continue

        # Rank relationships by absolute correlation
        sub['abs_rho'] = sub['rho'].abs()

        sorted_pairs = sub.sort_values(
            'abs_rho',
            ascending=False
        )

        pairs_to_plot = []

        # Select up to top_k valid relationships
        for _, row in sorted_pairs.iterrows():

            defect = row['Defect']
            factor = row['Factor']

            gcol = _resolve_group_col(
                factor,
                df,
                group_cols
            )

            if gcol is None:
                continue

            # Calculate grouped points
            pts = _points_for_plot_grouped(
                df,
                defect,
                gcol,
                id_col=id_col,
                defect_col=defect_col,
                material=mat,
                material_col=material_col,
                min_pipes=min_pipes_per_group,
                low_n_threshold=low_n_threshold
            )

            # Skip relationships with too few groups
            if (
                pts.empty
                or len(pts) < min_groups_to_show
            ):
                continue

            # Check numerical values
            x = pts['x_plot'].to_numpy(
                dtype=float
            )

            y = pts['mean_defects'].to_numpy(
                dtype=float
            )

            if not (
                np.isfinite(x).all()
                and np.isfinite(y).all()
            ):
                continue

            # Skip if there is no variation in x
            if np.ptp(x) == 0:
                continue

            pairs_to_plot.append(
                (row, gcol, pts)
            )

            if len(pairs_to_plot) == top_k:
                break

        # Draw selected relationships
        for i_r in range(n_rows):

            ax = axes[i_r, j]

            # Hide unused panels
            if i_r >= len(pairs_to_plot):
                ax.axis('off')
                continue

            row, gcol, pts = pairs_to_plot[i_r]

            defect = row['Defect']
            factor = row['Factor']
            rho = row['rho']
            pval = row['p']

            # Separate low-sample and valid groups
            pts_low = pts[
                pts['low_n']
            ]

            pts_ok = pts[
                ~pts['low_n']
            ]

            # Select material color
            if color_by_material:
                base_color = colors_materials[mat]

            else:
                base_color = 'steelblue'

            grey_color = '0.6'

            # Plot low-sample groups
            if not pts_low.empty:
                ax.scatter(
                    pts_low['x_plot'],
                    pts_low['mean_defects'],
                    s=40,
                    color=(
                        grey_color
                        if use_grey
                        else base_color
                    ),
                    label=f"<{low_n_threshold} pipes"
                )

            # Plot groups meeting the sample-size threshold
            if not pts_ok.empty:
                ax.scatter(
                    pts_ok['x_plot'],
                    pts_ok['mean_defects'],
                    s=40,
                    color=base_color,
                    label=None
                )

            # Store legend handles only once
            if (
                legend_handles is None
                and not pts_low.empty
            ):

                handles, labels = (
                    ax.get_legend_handles_labels()
                )

                if handles:
                    legend_handles = handles
                    legend_labels = labels

            # Optionally display values above points
            if show_point_labels:

                for _, point in pts.iterrows():

                    ax.text(
                        float(point['x_plot']),
                        float(point['mean_defects']),
                        f"{point['mean_defects']:.2f}",
                        fontsize=8,
                        ha='center',
                        va='bottom'
                    )

            # Select points used in the regression
            if fit_mode.lower() == "exclude":

                fit_df = pts_ok.copy()

            else:

                fit_df = pts.copy()

            # Format factor name
            if factor_renamer is not None:

                factor_display = (
                    factor_renamer.get(
                        factor,
                        factor
                    )
                )

            else:

                factor_display = factor

            base_title = f"{defect} vs {factor_display}"
            title = textwrap.fill(base_title, width=28)

            # Minimum number of points required for regression
            min_fit_points = max(
                min_groups_for_fit,
                2
            )

            # Fit regression when enough valid groups are available
            if len(fit_df) >= min_fit_points:

                xfit = (
                    fit_df['x_plot']
                    .astype(float)
                    .to_numpy()
                )

                yfit = (
                    fit_df['mean_defects']
                    .astype(float)
                    .to_numpy()
                )

                # Check regression input
                valid_fit = (
                    np.isfinite(xfit).all()
                    and np.isfinite(yfit).all()
                    and np.ptp(xfit) > 0
                )

                if valid_fit:

                    lr = linregress(
                        xfit,
                        yfit
                    )

                    xx = np.linspace(
                        np.nanmin(xfit),
                        np.nanmax(xfit),
                        100
                    )

                    yy = (
                        lr.intercept
                        + lr.slope * xx
                    )

                    # Plot regression line
                    ax.plot(
                        xx,
                        yy,
                        color=base_color,
                        linewidth=2
                    )

                    title += (
                        f"\nslope={lr.slope:.2g}, "
                        f"R²={lr.rvalue**2:.2f}"
                    )

                    results.append({
                        'MATERIAL': mat,
                        'Defect': defect,
                        'Factor': factor,
                        'group_col': gcol,
                        'rho': rho,
                        'p': pval,
                        'slope': lr.slope,
                        'intercept': lr.intercept,
                        'R2': lr.rvalue**2,
                        'n_groups_total': int(
                            len(pts)
                        ),
                        'n_groups_fit': int(
                            len(fit_df)
                        ),
                        'fit_mode': fit_mode
                    })

                else:

                    title += (
                        "\nNo regression "
                        "(insufficient x variation)"
                    )

            else:

                title += (
                    f"\nNo regression "
                    f"(n < {min_fit_points})"
                )

            # Format x-axis label
            if factor_renamer:

                x_label = factor_renamer.get(
                    factor,
                    factor
                )

            else:

                x_label = factor

            # Add material label to the first row
            if i_r == 0:

                ax.text(
                    0.5,
                    1.65,
                    rf"$\mathbf{{{mat}}}$",
                    transform=ax.transAxes,
                    ha='center',
                    va='center',
                    fontsize=20
                )

                ax.set_title(
                    title,
                    fontsize=18,
                    pad=18
                )

            else:

                ax.set_title(
                    title,
                    fontsize=18,
                    pad=18
                )

            # Axis labels
            ax.set_xlabel(
                x_label,
                fontsize=18
            )

            if j == 0:

                ax.set_ylabel(
                    "Mean defects per pipe",
                    fontsize=18
                )

            else:

                ax.set_ylabel("")

            # Format subplot borders
            for spine in ax.spines.values():

                spine.set_edgecolor(
                    "0.8"
                )

                spine.set_linewidth(
                    1.05
                )

            # Add grid
            ax.grid(
                True,
                ls='--',
                alpha=0.3
            )



    # Add one legend above the plots at the upper-right
    if legend_handles is not None:

        fig.legend(
            legend_handles,
            legend_labels,
            loc='upper right',
            bbox_to_anchor=(0.99, 0.995),
            fontsize=20,
            markerscale=2,
            frameon=True
        )

        # Reserve space above the subplot grid for the legend
        plt.tight_layout(
            rect=[0, 0, 1, 0.92]
        )

    else:

        plt.tight_layout()

    plt.show()

    return fig




