import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import linregress,spearmanr, pearsonr
from Config import build_material_color_map
from typing import List, Optional, Dict, Tuple, Mapping
from statsmodels.stats.multitest import multipletests


def defect_total_material(
    df: pd.DataFrame,
    id_col: str = "Pipe_ID",
    factors: Optional[List[str]] = None,
    min_samples: int = 15,
    method: str = "spearman",
    material: Optional[str] = None,
    material_col: str = "Material",apply_mtc: bool = True,p_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Correlation between pipe-level factors and total number of defects per pipe.
    One row = one pipe, total defects = count of defect records.
    Compute per-factor correlations for total defects for a given material.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing pipe-level aggregated defect metrics.
    id_col : str, default "Pipe_ID"
        Column identifying pipes.
    factors : list of str, optional
        List of factor columns to evaluate. If None, all numeric factors are used.
    min_samples : int, default 15
        Minimum number of samples per material required to compute correlations.
    method : {"spearman", "pearson"}, default "spearman"
        Correlation method to use.
    material : str, optional
        If provided, correlations are computed only for this material.
    material_col : str, default "Material"
        Name of the material column.

    Returns:
        rho_series, pval_series, signif_series, reason_series, n_pipes
    """
    # 1. Filter by material
    if material is None:
        df_mat = df.copy()
    elif isinstance(material, str):
        df_mat = df[df[material_col] == material].copy()
    else:
        df_mat = df[df[material_col].isin(material)].copy()

    # 2. Check required columns
    needed = {id_col} | set(factors or [])
    missing = needed - set(df_mat.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 3. Pipe-level factors (one row per pipe)
    pipe_factors = (
        df_mat[[id_col] + list(factors or [])]
        .drop_duplicates(subset=id_col)
        .set_index(id_col)
        .apply(pd.to_numeric, errors="coerce")
    )

    # 4. Total defects per pipe
    defect_counts = df_mat.groupby(id_col).size()
    defect_counts.name = "Total_defects"

    # 5. Align both tables
    common = pipe_factors.index.intersection(defect_counts.index)
    pipe_factors = pipe_factors.loc[common]
    defect_counts = defect_counts.loc[common]

    # 6. Minimum sample size
    if len(defect_counts) < min_samples:
        raise ValueError(f"Only {len(defect_counts)} pipes after filtering (need ≥ {min_samples})")

    # 7. Correlation loop
    rho_series = pd.Series(index=factors, dtype=float)
    pval_series = pd.Series(index=factors, dtype=float)
    reason_series = pd.Series(index=factors, dtype=object)

    y = defect_counts.astype(float)
    use_spearman = method.lower() == "spearman"

    for factor in factors:

        x = pipe_factors[factor]
        mask = x.notna() & y.notna()
        n_valid = int(mask.sum())

        if n_valid < min_samples:
            rho = np.nan
            p_value = np.nan
            reason = f"Only {n_valid} valid pairs"

        elif x.loc[mask].nunique() <= 1:
            rho = np.nan
            p_value = np.nan
            reason = "Factor constant"

        elif y.loc[mask].nunique() <= 1:
            rho = np.nan
            p_value = np.nan
            reason = "Defects constant"

        else:
            if use_spearman:
                rho, p_value = spearmanr(
                    x.loc[mask],
                    y.loc[mask],
                )
            else:
                rho, p_value = pearsonr(
                    x.loc[mask],
                    y.loc[mask],
                )

            reason = "OK"

        rho_series.loc[factor] = rho
        pval_series.loc[factor] = p_value
        reason_series.loc[factor] = reason

    # --------------------------------------------------------------
    # Multiple-testing correction after evaluating all factors
    # --------------------------------------------------------------

    pval_adj_series = pd.Series(
        np.nan,
        index=pval_series.index,
        dtype=float,
    )

    valid_pvalues = pval_series.dropna()

    if not valid_pvalues.empty:

        if apply_mtc:
            _, corrected_pvalues, _, _ = multipletests(
                valid_pvalues.to_numpy(),
                alpha=p_threshold,
                method="fdr_bh",
            )

            pval_adj_series.loc[valid_pvalues.index] = (
                corrected_pvalues
            )

        else:
            pval_adj_series.loc[valid_pvalues.index] = (
                valid_pvalues
            )

    signif_series = pd.Series(
        index=factors,
        dtype=object,
    )

    for factor in factors:

        rho = rho_series.loc[factor]
        adjusted_p = pval_adj_series.loc[factor]

        if pd.isna(rho):
            signif_series.loc[factor] = "nan"

        elif (
            pd.notna(adjusted_p)
            and adjusted_p < p_threshold
        ):
            signif_series.loc[factor] = f"{rho:.2f}*"

        else:
            signif_series.loc[factor] = f"{rho:.2f}"

    return (
        rho_series,
        pval_adj_series,
        signif_series,
        reason_series,
        len(defect_counts),
    )


# Helper: midpoint from group label
def _mid_from_label(g):
    if isinstance(g, pd.Interval):
        return (g.left + g.right) / 2.0
    s = str(g).replace("–", "-").replace("—", "-").strip()
    if m := re.match(r"^(\d+(?:\.\d+)?)\s*\+\s*$", s):
        return float(m.group(1))
    nums = re.findall(r"\d+(?:\.\d+)?", s)
    if len(nums) >= 2:
        return (float(nums[0]) + float(nums[1])) / 2.0
    if len(nums) == 1:
        return float(nums[0])
    return np.nan


# Resolve grouped column
def _resolve_group_col_for_factor(
    factor_raw: str,
    df: pd.DataFrame,
    group_suffix: str = "_group",
):
    """
    Resolve the grouped column name for a given raw factor.

    Example:
        factor_raw = "Slope" → returns "Slope_group" if in df.columns.

    Parameters
    ----------
    factor_raw : str
        Base name of the factor (e.g., "Slope").
    df : pd.DataFrame
        DataFrame that may contain grouped columns.
    group_suffix : str, default "_group"
        Suffix appended to the raw factor to form the grouped column name.
    """

    cand = f"{factor_raw}{group_suffix}"

    if cand in df.columns:
        return cand

    return None


def compute_points_total_by_group(
    df_grouped: pd.DataFrame,
    material: str,
    factor_raw: str,
    id_col: str = "Pipe_ID",
    material_col: str = "Material",
    min_pipes_per_group: int = 1,
    min_samples: int = 15,
    group_suffix: str = "_group"
) -> pd.DataFrame:
    """
    Returns a DataFrame ready for plotting: mean total defects vs factor (numeric or grouped).

    Parameters
    ----------
    df_grouped : pd.DataFrame
        Dataframe containing at least:
        - Pipe-level metrics (e.g., total defects)
        - Grouped factor column (factor_raw + group_suffix)
        - Material column
        - Pipe ID column
    material : str
        The material to filter the analysis on (e.g., "PVC").
    factor_raw : str
        Base name of the factor being analyzed. The grouped column is assumed
        to be `factor_raw + group_suffix`.
    id_col : str, default "Pipe_ID"
        Column identifying pipe-level unique IDs.
    material_col : str, default "Material"
        Column identifying pipe material.
    min_pipes_per_group : int, default 1
        Minimum number of pipes required for a group to be included.
    min_samples : int, default 15
        Groups below this count are flagged as low sample size.
    group_suffix : str, default "_group"
        Suffix used to identify the grouped factor column.
    """
    sub = df_grouped[df_grouped[material_col] == material].copy()
    if sub.empty:
        return pd.DataFrame(columns=["group", "x_plot", "mean_total_defects", "n_pipes", "low_n"])

    # Total defects per pipe
    total_per_pipe = sub.groupby(id_col).size().rename("Total_defects")

    # Columns we need
    cols = [id_col]
    if factor_raw in sub.columns:
        cols.append(factor_raw)
    gcol = _resolve_group_col_for_factor(factor_raw, sub, group_suffix)
    if gcol:
        cols.append(gcol)

    if not set(cols).issubset(sub.columns):
        return pd.DataFrame(columns=["group", "x_plot", "mean_total_defects", "n_pipes", "low_n"])

    # Pipe-level table
    base = sub[cols].drop_duplicates(subset=id_col).set_index(id_col)
    merged = base.join(total_per_pipe, how="left").reset_index()
    merged["Total_defects"] = merged["Total_defects"].fillna(0)

    if gcol:  # grouped version exists → use groups
        agg = (
            merged.groupby(gcol, dropna=False, observed=False)
            .agg(mean_total_defects=("Total_defects", "mean"), n_pipes=(id_col, "nunique"))
            .reset_index()
            .rename(columns={gcol: "group"})
        )
        agg["x_plot"] = agg["group"].apply(_mid_from_label)
    else:
        if factor_raw not in merged.columns:
            return pd.DataFrame(columns=["group", "x_plot", "mean_total_defects", "n_pipes", "low_n"])
        agg = (
            merged.groupby(factor_raw, dropna=False)
            .agg(
                mean_total_defects=("Total_defects", "mean"),
                n_pipes=(id_col, "nunique"),
                x_plot=(factor_raw, "mean"),
            )
            .reset_index()
            .rename(columns={factor_raw: "group"})
        )

    agg = agg[agg["n_pipes"] >= min_pipes_per_group].copy()
    agg["low_n"] = agg["n_pipes"] < min_samples
    agg = agg.dropna(subset=["x_plot"]).sort_values("x_plot").reset_index(drop=True)

    return agg[["group", "x_plot", "mean_total_defects", "n_pipes", "low_n"]]


def correlations_by_material_from_defect_total(
    df: pd.DataFrame,
    factors: List[str],
    materials: Optional[List[str]] = None,
    id_col: str = "Pipe_ID",
    material_col: str = "Material",
    min_samples: int = 30,
    method: str = "spearman",
    apply_mtc:bool = True,
    p_threshold:float = 0.05,
) -> pd.DataFrame:
    """
    Run defect_total_material for every material → long-format table.

    Parameters
    ----------
    df : pd.DataFrame
        Pipe-level dataframe. Must contain:
        - A defect metric column (e.g., 'DEFECTS_PER_PIPE' or aggregated version)
        - Factor columns listed in `factors`
        - Material column
        - Pipe ID column
    factors : list of str
        List of factor column names to compute correlations against.
    materials : list of str, optional
        Materials to include. If None, materials present in `df[material_col]`
        are used.
    id_col : str, default "Pipe_ID"
        Column identifying pipe-level unique IDs.
    material_col : str, default "Material"
        Column identifying pipe material.
    min_samples : int, default 15
        Minimum number of samples required within each material for correlation.
    method : {"spearman", "pearson"}, default "spearman"
        Correlation method to use.
    """
    if materials is None:
        materials = sorted(df[material_col].dropna().unique())

    rows = []
    for mat in materials:
        rho_s, p_s, _, _, n = defect_total_material(
            df=df,
            id_col=id_col,
            factors=factors,
            min_samples=min_samples,
            method=method,
            material=mat,
            material_col=material_col,
            apply_mtc=apply_mtc,
            p_threshold=p_threshold
        )
        for f in factors:
            rows.append(
                {
                    "Material": mat,
                    "Factor": f,
                    "rho": float(rho_s.get(f, np.nan)),
                    "p": float(p_s.get(f, np.nan)),
                    "n": int(n),
                }
            )
    return pd.DataFrame(rows)


def plot_topk_total_by_material(
    df_grouped: pd.DataFrame,
    corr_df_material: pd.DataFrame,
    materials_order: Optional[List[str]] = None,
    selected_materials: Optional[List[str]] = None,
    factors: List[str] = None,
    p_thr: float = 0.05,
    id_col: str = "Pipe_ID",
    material_col: str = "Material",
    min_pipes_per_group: int = 1,
    min_samples: int = 30,
    min_groups_for_fit: int = 3,
    fit_mode: str = "include",
    cell_size: Tuple[float, float] = (5.0, 3.8),
    show_point_labels: bool = False,
    display_names: Mapping[str, str] | None = None,
):
    """
    Plot the same fixed factors across selected materials.
    - Same X and Y scale per factor (row) for comparison.
    - Only plots cells that are significant AND can perform a fit.
    - Materials with no significant factors are excluded.
    """
    # --------------------------------------------------------------
    # 1. Validate inputs
    # --------------------------------------------------------------
    required_columns = {"Material", "Factor", "rho", "p"}
    missing = required_columns - set(corr_df_material.columns)
    if missing:
        raise ValueError(f"Missing columns in corr_df_material: {', '.join(sorted(missing))}")

    if fit_mode not in {"include", "exclude"}:
        raise ValueError("fit_mode must be either 'include' or 'exclude'.")

    if factors is None:
        factors = ["Installation_year", "Pipe_length", "Properties"]

    # --------------------------------------------------------------
    # 2. Define materials to plot
    # --------------------------------------------------------------
    if selected_materials is not None:
        candidates = list(selected_materials)
    elif materials_order is not None:
        candidates = list(materials_order)
    else:
        candidates = (
            corr_df_material["Material"]
            .dropna()
            .drop_duplicates()
            .tolist()
        )

    # --------------------------------------------------------------
    # 3. Significant results
    # --------------------------------------------------------------
    significant_results = corr_df_material.loc[
        corr_df_material["p"].notna()
        & corr_df_material["rho"].notna()
        & (corr_df_material["p"] < p_thr)
    ].copy()

    # --------------------------------------------------------------
    # 4. Pre-compute points + decide which materials really have something to plot
    # --------------------------------------------------------------
    points_cache = {}
    global_xlim = {}
    global_ylim = {}

    # First pass: compute points for all candidates
    for factor in factors:
        for material in candidates:
            points = compute_points_total_by_group(
                df_grouped=df_grouped,
                material=material,
                factor_raw=factor,
                id_col=id_col,
                material_col=material_col,
                min_pipes_per_group=min_pipes_per_group,
                min_samples=min_samples,
                group_suffix="_group",
            )
            points_cache[(material, factor)] = points

    # Second pass: keep only materials that have at least one plottable cell
    materials_to_plot = []
    for material in candidates:
        has_plottable = False
        for factor in factors:

            mask = (
                (significant_results["Material"] == material)
                & (significant_results["Factor"] == factor)
            )
            if mask.any() is False:
                continue

            points = points_cache[(material, factor)]
            if points.empty or len(points) < 2:
                continue

            if fit_mode == "exclude":
                fit_df = points.loc[~points["low_n"]]
            else:
                fit_df = points

            can_fit = (
                len(fit_df) >= min_groups_for_fit
                and fit_df["x_plot"].nunique() >= 2
            )
            if can_fit:
                has_plottable = True
                break

        if has_plottable:
            materials_to_plot.append(material)

    if not materials_to_plot:
        print("No material with significant regressions that can be fitted.")
        return None, pd.DataFrame()

    # Now compute global scales only with the final materials
    for factor in factors:
        all_x, all_y = [], []
        for material in materials_to_plot:
            points = points_cache[(material, factor)]
            if not points.empty:
                all_x.extend(points["x_plot"].dropna().tolist())
                all_y.extend(points["mean_total_defects"].dropna().tolist())

        if all_x:
            global_xlim[factor] = (min(all_x), max(all_x))
            global_ylim[factor] = (min(all_y), max(all_y))
        else:
            global_xlim[factor] = (0, 1)
            global_ylim[factor] = (0, 1)

    # --------------------------------------------------------------
    # 5. Layout
    # --------------------------------------------------------------
    n_rows = len(factors)
    n_cols = len(materials_to_plot)

    material_colors = build_material_color_map(materials_to_plot)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_size[0], n_rows * cell_size[1]),
        squeeze=False,
    )

    def _mcolor(material: str):
        return material_colors.get(material, None)

    summaries = []

    # --------------------------------------------------------------
    # 6. Plot
    # --------------------------------------------------------------
    for col_idx, material in enumerate(materials_to_plot):

        axes[0, col_idx].set_title(
            material, fontsize=22, pad=42, fontweight="bold"
        )

        for row_idx, factor_raw in enumerate(factors):

            ax = axes[row_idx, col_idx]
            points = points_cache[(material, factor_raw)]

            factor_name = (
                display_names.get(factor_raw, factor_raw)
                if isinstance(display_names, Mapping)
                else factor_raw
            )

            # 1. No points → turn off axis
            if points.empty or len(points) < 2:
                ax.axis("off")
                continue

            # 2. Is it significant?
            mask = (
                (significant_results["Material"] == material)
                & (significant_results["Factor"] == factor_raw)
            )
            result_rows = significant_results.loc[mask]
            is_significant = not result_rows.empty

            if not is_significant:
                ax.axis("off")
                continue

            # 3. Prepare data for the fit
            if fit_mode == "exclude":
                fit_df = points.loc[~points["low_n"]].copy()
            else:
                fit_df = points.copy()

            # 4. Can we perform the fit?
            can_fit = (
                len(fit_df) >= min_groups_for_fit
                and fit_df["x_plot"].nunique() >= 2
            )

            if not can_fit:
                ax.axis("off")
                continue

            # ----------------------------------------------------------
            # From here on we DO plot (significant + fit is possible)
            # ----------------------------------------------------------
            result = result_rows.iloc[0]
            rho_raw = result["rho"]
            p_raw = result["p"]

            # Scatter
            points_low = points.loc[points["low_n"]]
            points_ok = points.loc[~points["low_n"]]

            if not points_low.empty:
                ax.scatter(
                    points_low["x_plot"],
                    points_low["mean_total_defects"],
                    s=45,
                    alpha=0.9,
                    c="0.6",
                )

            if not points_ok.empty:
                ax.scatter(
                    points_ok["x_plot"],
                    points_ok["mean_total_defects"],
                    s=45,
                    alpha=0.9,
                    c=_mcolor(material),
                )

            if show_point_labels:
                for _, point in points.iterrows():
                    ax.text(
                        float(point["x_plot"]),
                        float(point["mean_total_defects"]),
                        f"{point['mean_total_defects']:.2f}",
                        fontsize=14, ha="center", va="bottom",
                    )

            # Regression
            x = fit_df["x_plot"].astype(float).to_numpy()
            y = fit_df["mean_total_defects"].astype(float).to_numpy()
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]

            regression = linregress(x, y)
            x_line = np.linspace(np.min(x), np.max(x), 100)
            y_line = regression.intercept + regression.slope * x_line

            ax.plot(x_line, y_line, linewidth=2, c=_mcolor(material))

            subtitle_text = (
                f"Slope={regression.slope:.2g}, "
                f"R²={regression.rvalue**2:.2f}"
            )

            summaries.append({
                "Material": material,
                "Factor": factor_raw,
                "rho_raw": rho_raw,
                "p_adjusted": p_raw,
                "slope_grouped": regression.slope,
                "R2_grouped": regression.rvalue**2,
                "n_groups_total": int(len(points)),
                "n_groups_fit": int(len(fit_df)),
                "fit_mode": fit_mode,
            })

            # Subtitle: slope and R² above the plot
            ax.text(
                0.5,
                1.02,
                subtitle_text,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=18,
                fontweight="normal",
                clip_on=False,
            )

            # Common scales
            ax.set_xlim(global_xlim[factor_raw])
            ax.set_ylim(global_ylim[factor_raw])

            # Axes
            ax.set_xlabel(factor_name, fontsize=18)
            if col_idx == 0:
                ax.set_ylabel("Mean defects per pipe", fontsize=18)
            else:
                ax.set_ylabel("")

            ax.grid(True, linestyle="--", alpha=0.3)
            ax.tick_params(axis="both", labelsize=16)

            for spine in ax.spines.values():
                spine.set_edgecolor("0.8")
                spine.set_linewidth(1.05)

    # --------------------------------------------------------------
    # Global legend
    # --------------------------------------------------------------
    low_n_legend = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="None",
        markerfacecolor="0.6",
        markeredgecolor="0.6",
        markersize=14,
        label=f"< {min_samples} pipes",
    )

    fig.legend(
        handles=[low_n_legend],
        loc="upper right",
        bbox_to_anchor=(0.94, 1.05),
        frameon=True,
        fontsize=20,
    )
    plt.tight_layout(
        rect=[0, 0, 0.94, 0.97],
        h_pad=3.0,
        w_pad=2.0,
    )
    plt.show()

    return fig, pd.DataFrame(summaries)