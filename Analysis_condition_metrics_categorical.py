from typing import Mapping, Sequence
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from matplotlib.ticker import MaxNLocator
import math
from Config import get_categories_order


def analyze_categorical_kw(
        df_cctv_filtered,
        df_defect,
        id_col="Pipe_ID",
        material_col="Material",
        selected_materials=None,
        length_col="Pipe_length",
        categorical_vars=None,
        metric_cols=["Defects_per_pipe", "Condition_score"],
        condition:str="Conidition_score",
        metric_titles=[
            "Defects per pipe",
            "Condition score"
        ],
        display_names=None,
        alpha=0.05,
        min_epsilon2=0.06,
        show=False
):
    """
    Full Kruskal–Wallis analysis with FDR per metric and material:
    - Computes defect metrics per pipe
    - Builds H-statistic matrices per metric and material
    - Produces 1×N heatmap grid (one heatmap per metric)
    - Cell color = H statistic, text = H
    - Non-significant cells shown as light grey

    Parameters
    ----------
    df_cctv_filtered : DataFrame
        Pipe-level CCTV dataframe (one row per pipe).
    df_defect : DataFrame
        Defect-level dataframe (one row per defect).
    selected_materials : list-like or None
        List of materials to include. If None, uses all materials in df_cctv_filtered.
    id_col : str
        Pipe ID column name.
    material_col : str
        Material column name.
    length_col : str
        Pipe length column name (in meters).
    categorical_vars : list of str
        Categorical variables to analyze (e.g. land cover, land use, soil type).
    metric_cols : list of str
        Metric columns in df_pipes to analyze.
    metric_titles : list of str
        Titles for each metric in plots.
    display_names : dict
        Mapping of column name to display name for plotting.
    alpha : float
        Significance level for Kruskal–Wallis test.
    """

    if categorical_vars is None:
        raise ValueError("categorical_vars must be provided")

    if metric_titles is None:
        metric_titles = metric_cols

    display_names = display_names or {f: f.replace('_', ' ') for f in categorical_vars}

    # ----------------------------------------------------------
    # 1. Construct pipe-level dataframe with metrics
    # ----------------------------------------------------------

    cols_needed = [id_col, material_col, length_col, condition] + categorical_vars
    df_pipes = df_cctv_filtered[cols_needed].copy()

    # Defects per pipe
    defect_counts = (
        df_defect.groupby([id_col, material_col])
        .size()
        .reset_index(name="Defects_per_pipe")
    )
    df_pipes = df_pipes.merge(defect_counts, on=[id_col, material_col], how="left")
    df_pipes["Defects_per_pipe"] = df_pipes["Defects_per_pipe"].fillna(0)

    # Defects per km
    df_pipes[length_col] = pd.to_numeric(df_pipes[length_col], errors="coerce")
    df_pipes["Defects_per_km"] = df_pipes["Defects_per_pipe"] / (df_pipes[length_col] / 1000.0)

    # Materials to plot
    if selected_materials is None:
        materials = list(df_pipes[material_col].dropna().unique())
    else:
        available = [m for m in selected_materials if m in df_pipes[material_col].unique()]

        materials = list(dict.fromkeys(
            m for m in selected_materials if m in available
        ))

    # ----------------------------------------------------------
    # 2. Initialize H , p-value matrices, and n matrix
    # ----------------------------------------------------------
    n_matrix = {metric: pd.DataFrame(index=categorical_vars, columns=materials, dtype=float)
                for metric in metric_cols}
    heatmap_H = {metric: pd.DataFrame(index=categorical_vars, columns=materials, dtype=float)
                 for metric in metric_cols}
    pval_matrix = {metric: pd.DataFrame(index=categorical_vars, columns=materials, dtype=float)
                   for metric in metric_cols}
    H_matrix = {metric: pd.DataFrame(index=categorical_vars, columns=materials, dtype=float)
                for metric in metric_cols}

    # ----------------------------------------------------------
    # 2.1. Epsilon
    # ----------------------------------------------------------

    n_matrix = {metric: pd.DataFrame(index=categorical_vars, columns=materials, dtype=float)
                for metric in metric_cols}

    heatmap_H = {metric: pd.DataFrame(index=categorical_vars, columns=materials, dtype=float)
                 for metric in metric_cols}

    pval_matrix = {metric: pd.DataFrame(index=categorical_vars, columns=materials, dtype=float)
                   for metric in metric_cols}

    H_matrix = {metric: pd.DataFrame(index=categorical_vars, columns=materials, dtype=float)
                for metric in metric_cols}

    epsilon_matrix = {
        metric: pd.DataFrame(index=categorical_vars, columns=materials, dtype=float)
        for metric in metric_cols
    }

    # ----------------------------------------------------------
    # 3. Compute Kruskal–Wallis per metric & material
    # ----------------------------------------------------------
    for metric in metric_cols:
        for mat in materials:
            sub = df_pipes[df_pipes[material_col] == mat]

            pvals_list = []
            valid_factors = []

            for cat in categorical_vars:
                levels = sub[cat].dropna().unique()
                if len(levels) < 2:
                    pval_matrix[metric].loc[cat, mat] = np.nan
                    H_matrix[metric].loc[cat, mat] = np.nan
                    continue

                groups = [grp[metric].dropna() for _, grp in sub.groupby(cat) if len(grp[metric].dropna()) > 0]

                if len(groups) < 2:
                    pval_matrix[metric].loc[cat, mat] = np.nan
                    H_matrix[metric].loc[cat, mat] = np.nan
                    continue

                try:
                    H, p = kruskal(*groups)
                except:
                    pval_matrix[metric].loc[cat, mat] = np.nan
                    H_matrix[metric].loc[cat, mat] = np.nan
                    n_matrix[metric].loc[cat, mat] = np.nan
                    continue

                # Store raw H, p and n
                n = sum(len(g) for g in groups)
                H_matrix[metric].loc[cat, mat] = H
                pval_matrix[metric].loc[cat, mat] = p
                n_matrix[metric].loc[cat, mat] = n
                pvals_list.append(p)
                valid_factors.append(cat)

            # Apply multiple testing correction per metric + material
            if len(pvals_list) > 0:
                reject, pvals_corrected, _, _ = multipletests(pvals_list, alpha=alpha, method='fdr_bh')
                for cat, p_corr in zip(valid_factors, pvals_corrected):
                    pval_matrix[metric].loc[cat, mat] = p_corr

            # Fill final H matrix masking non-significant cells and no practical meaning
            for cat in categorical_vars:
                p_corr = pval_matrix[metric].loc[cat, mat]
                H_val = H_matrix[metric].loc[cat, mat]
                n_val = n_matrix[metric].loc[cat, mat]

                if (pd.notna(p_corr) and p_corr < alpha and
                        pd.notna(H_val) and pd.notna(n_val) and n_val > 1):
                    epsilon2 = H_val / (n_val - 1)

                    epsilon_matrix[metric].loc[cat, mat] = epsilon2

                    if epsilon2 >= min_epsilon2:
                        heatmap_H[metric].loc[cat, mat] = H_val
                    else:
                        heatmap_H[metric].loc[cat, mat] = np.nan

                else:
                    heatmap_H[metric].loc[cat, mat] = np.nan
                    epsilon_matrix[metric].loc[cat, mat] = np.nan

    # ----------------------------------------------------------
    # 4. Plotting
    # ----------------------------------------------------------

    # Keep epsilon² only for cells that passed:
    #   1. Kruskal-Wallis significance after FDR correction
    #   2. minimum epsilon² threshold
    heatmap_epsilon = {
        metric: epsilon_matrix[metric].where(heatmap_H[metric].notna())
        for metric in metric_cols
    }

    # ----------------------------------------------------------
    # Common colour scale based on epsilon²
    # ----------------------------------------------------------
    all_epsilon = np.concatenate([
        heatmap_epsilon[m].values.flatten()
        for m in metric_cols
    ])

    finite_epsilon = all_epsilon[np.isfinite(all_epsilon)]

    if len(finite_epsilon) > 0:
        epsilon_max = finite_epsilon.max()
    else:
        epsilon_max = 1.0

    norm = mpl.colors.Normalize(
        vmin=0,
        vmax=epsilon_max
    )

    sns.set(style="white")

    fig, axes = plt.subplots(
        1,
        len(metric_cols),
        figsize=(4.8 * len(metric_cols), 0.8 * len(categorical_vars)),
        sharey=True
    )

    if len(metric_cols) == 1:
        axes = [axes]

    mappable = None

    for ax, metric, title in zip(axes, metric_cols, metric_titles):

        # ------------------------------------------------------
        # Data shown in heatmap
        # ------------------------------------------------------
        data_epsilon = heatmap_epsilon[metric].reindex(categorical_vars)

        # H values only for cells that passed the filters
        data_H = heatmap_H[metric].reindex(categorical_vars)

        # Display names
        if display_names:
            data_epsilon = data_epsilon.rename(index=display_names)
            data_H = data_H.rename(index=display_names)

        # Add dash after factor names
        data_epsilon.index = [f"{idx} -" for idx in data_epsilon.index]
        data_H.index = data_epsilon.index

        # Ensure material order
        data_epsilon = data_epsilon.reindex(columns=materials)
        data_H = data_H.reindex(columns=materials)

        # Mask cells that did not pass the filters
        mask = data_epsilon.isna()

        # ------------------------------------------------------
        # Custom annotations:
        # epsilon² on first line
        # H statistic in parentheses underneath
        # ------------------------------------------------------
        annotations = pd.DataFrame(
            "",
            index=data_epsilon.index,
            columns=data_epsilon.columns
        )

        for row in data_epsilon.index:
            for col in data_epsilon.columns:

                eps_val = data_epsilon.loc[row, col]
                H_val = data_H.loc[row, col]

                if pd.notna(eps_val) and pd.notna(H_val):
                    annotations.loc[row, col] = (
                        f"{eps_val:.3f}\n"
                        f"(H={H_val:.1f})"
                    )

        # ------------------------------------------------------
        # Heatmap
        # ------------------------------------------------------
        cmap = mpl.cm.Greens.copy()
        cmap.set_bad(color="lightgrey")

        hm = sns.heatmap(
            data_epsilon,
            ax=ax,
            cmap=cmap,
            mask=mask,
            annot=annotations,
            fmt="",
            linewidths=0.5,
            linecolor="white",
            cbar=False,
            norm=norm,
            annot_kws={
                "fontsize": 10,
                "ha": "center",
                "va": "center"
            }
        )

        if mappable is None:
            mappable = hm.collections[0]

        ax.set_title(
            title,
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.tick_params(
            axis="x",
            which="major",
            direction="out",
            length=2,
            width=0.8,
            bottom=True,
            top=False,
            rotation=0
        )

        ax.tick_params(
            axis="y",
            length=5,
            width=1,
            pad=0,
            rotation=0
        )

        for label in ax.get_xticklabels():
            label.set_horizontalalignment("center")


    # ----------------------------------------------------------
    # Shared colour bar
    # ----------------------------------------------------------
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(mappable, cax=cbar_ax)

    cbar.ax.set_ylabel(
        "Epsilon squared (ε²)",
        rotation=90,
        va="center",
        labelpad=18,
    )
    cbar.ax.yaxis.set_label_position("right")

    fig.subplots_adjust(right=0.84)
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, df_pipes, epsilon_matrix


def kruskal_defects_by_category(
        df_cctv_filtered: pd.DataFrame,
        df_defects: pd.DataFrame,
        categorical_vars: Sequence[str],
        metric_cols: Sequence[str] = ("Defects_per_pipe",),
        id_col: str = "Pipe_ID",
        material_col: str = "Material",
        alpha: float = 0.05
):
    """
    Calculate Defects_per_pipe and perform Kruskal–Wallis tests on defect counts
    across categories for a set of categorical variables and multiple metrics,
    applying FDR correction per metric and per material.

    Parameters
    ----------
    df_cctv_filtered : DataFrame of pipes
    df_defects       : DataFrame of individual defects
    categorical_vars : list of categorical columns to test
    metric_cols      : metrics to analyse (default: Defects_per_pipe)
    id_col           : column that uniquely identifies each pipe
    material_col     : material column
    alpha            : significance level for FDR

    Returns
    -------
    DataFrame with corrected p-values and significance conclusions.
    """

    # ----- Calculate Defects_per_pipe -----
    defect_counts = (
        df_defects.groupby([id_col, material_col])
        .size()
        .reset_index(name="Defects_per_pipe")
    )
    if "Defects_per_pipe" in df_cctv_filtered.columns:
        df_cctv_filtered = df_cctv_filtered.drop(columns=["Defects_per_pipe"])

    temp = df_cctv_filtered.merge(defect_counts, on=[id_col, material_col], how="left")
    df_cctv_filtered["Defects_per_pipe"] = temp["Defects_per_pipe"].fillna(0).values

    df_pipes = df_cctv_filtered
    results = []
    materials = df_pipes[material_col].dropna().unique()

    for metric_col in metric_cols:
        for mat in materials:
            df_mat = df_pipes[df_pipes[material_col] == mat]

            # First pass: collect p-values
            pvals_list = []
            valid_vars = []  # (var, H, raw_p)

            for var in categorical_vars:
                sub = df_mat[[var, metric_col]].dropna()
                categories = sub[var].unique()
                if len(categories) < 2 or sub.empty:
                    continue

                groups = [sub[sub[var] == c][metric_col].values for c in categories]
                if any(len(g) == 0 for g in groups):
                    continue

                stat, p = kruskal(*groups)
                pvals_list.append(p)
                valid_vars.append((var, stat, p))

            # FDR correction
            if len(pvals_list) > 0:
                reject, pvals_corrected, _, _ = multipletests(
                    pvals_list, alpha=alpha, method="fdr_bh"
                )

                for (var, stat, p_raw), p_corr, is_sig in zip(
                        valid_vars, pvals_corrected, reject
                ):
                    results.append({
                        "Material": mat,
                        "Metric": metric_col,
                        "Variable": var,
                        "Kruskal_stat": stat,
                        "p_value_raw": p_raw,
                        "p_value_corrected": p_corr,
                        "Significant": is_sig,
                        "Conclusion": (
                            f"Different (FDR p < {alpha:.3f})"
                            if is_sig
                            else "Not different"
                        ),
                    })

            # Variables that could not be tested
            skipped_vars = set(categorical_vars) - {v for v, _, _ in valid_vars}
            for var in skipped_vars:
                results.append({
                    "Material": mat,
                    "Metric": metric_col,
                    "Variable": var,
                    "Kruskal_stat": None,
                    "p_value_raw": None,
                    "p_value_corrected": None,
                    "Significant": False,
                    "Conclusion": "Not enough categories or empty group",
                })

    return pd.DataFrame(results)


def plot_boxplots_by_material(
        df: pd.DataFrame,
        cat_vars: Sequence[str],
        significant_df=pd.DataFrame,
        selected_materials: Sequence[str] = None,
        boxplots_labels: Mapping[str, str] = None,
        min_samples: int = 30,
        categories_colors: Mapping[str, Mapping[str, str]] = None,
        categories_order: Mapping[str, Sequence[str]] = None,
):
    # Filter for selected materials
    df = df[df["Material"].isin(selected_materials)].copy()

    n_rows = len(cat_vars)
    n_cols = len(selected_materials)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 2.5 * n_rows), sharey=False)

    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Legend reference for low-count categories
    grey_patch = mpatches.Patch(color='grey', label=f'Categories with < {min_samples} records')

    # Legend reference for Kruskal–Wallis significance
    red_border = Line2D([0], [0], color='red', linewidth=3, label='Significant in Kruskal–Wallis')

    legend_added = False

    # Font sizes
    title_fontsize = 18
    label_fontsize = 20
    tick_fontsize = 14
    legend_fontsize = 20

    for i, var in enumerate(cat_vars):

        order = None
        if categories_order and var in categories_order:
            order = categories_order[var]

        for j, mat in enumerate(selected_materials):
            ax = axes[i, j]

            subset = df[df["Material"] == mat].dropna(subset=[var])
            if subset.empty:
                ax.set_visible(False)
                continue

            subset[var] = subset[var].astype(str)
            counts = subset[var].value_counts()
            valid_cats = counts[counts >= min_samples].index.astype(str)
            all_cats = counts.index.astype(str)

            if categories_colors and var in categories_colors:
                palette_dict = categories_colors[var].copy()
            else:
                colors = sns.color_palette("Set2", n_colors=len(all_cats))
                palette_dict = dict(zip(all_cats, colors))

            for cat in all_cats:
                if cat not in valid_cats:
                    palette_dict[cat] = "grey"

            sns.set_context("paper")
            sns.boxplot(
                data=subset,
                x=var,
                y="Defects_per_pipe",
                hue=var,
                legend=False,
                order=order,
                palette=palette_dict,
                showfliers=False,
                ax=ax
            )
            ax.yaxis.set_major_locator(
                MaxNLocator(
                    nbins=7,
                    integer=True
                )
            )
            ax.tick_params(
                axis="x",
                which="major",
                direction="out",
                length=6,
                width=0.8,
                bottom=True,
                top=False,
                labelsize=tick_fontsize,
                rotation=45
            )

            ax.tick_params(
                axis="y",
                which="major",
                direction="out",
                length=4,
                width=1,
                left=True,
                right=False,
                labelsize=tick_fontsize
            )
            # Title
            if i == 0:
                ax.set_title(mat, fontsize=title_fontsize, fontweight="bold")

            # Kruskal–Wallis significance
            is_significant = False
            mask = (significant_df['Material'] == mat) & (significant_df['Variable'] == var)
            if mask.any():
                is_significant = bool(significant_df.loc[mask, 'Significant'].values[0])

            if is_significant:
                for spine in ax.spines.values():
                    spine.set_color('red')
                    spine.set_linewidth(2.5)
                    spine.set_visible(True)

            if j == 0:
                label = boxplots_labels.get(var, var)
                ax.set_ylabel(f"{label}\n\nDefects per pipe", fontsize=label_fontsize)
            else:
                ax.set_ylabel("")

            ax.set_xlabel("")
            # ax.tick_params(axis="x", labelsize=tick_fontsize, rotation=45)
            # ax.tick_params(axis="y", labelsize=tick_fontsize)

            # Add global legend once
            if not legend_added:
                fig.legend(
                    handles=[grey_patch, red_border],
                    loc='lower left',
                    bbox_to_anchor=(0, -0.1),
                    fontsize=legend_fontsize,
                    ncol=1,
                    frameon=False
                )
                legend_added = True

    plt.tight_layout()
    return fig

def plot_boxplots_high_epsilon(
        df_pipes,
        epsilon_matrix,
        categorical_vars,
        metric_cols,
        selected_materials=None,
        colors_materials=None,
        material_col="Material",
        min_epsilon2=0.12,
        display_names=None,
        metric_titles=None,
        ncols=1,
        figsize_per_plot=(2.5, 2),
        show=False
):
    """
    Plot boxplots for Kruskal-Wallis results with epsilon squared
    greater than or equal to a specified threshold.

    Parameters
    ----------
    df_pipes : pd.DataFrame
        Pipe-level dataframe used in the Kruskal-Wallis analysis.

    epsilon_matrix : dict
        Dictionary containing one epsilon² matrix per metric.
        Rows = categorical variables
        Columns = materials.

    categorical_vars : list
        Categorical variables analyzed.

    metric_cols : list
        Metrics analyzed.

    selected_materials : list or None
        Materials to include.

    material_col : str
        Material column name.

    min_epsilon2 : float
        Minimum epsilon squared to plot.

    display_names : dict or None
        Display names for categorical variables.

    metric_titles : list or None
        Display names for metrics.

    ncols : int
        Number of columns in subplot grid.

    figsize_per_plot : tuple
        Width and height per subplot.
    """

    categories_order = get_categories_order()

    # ----------------------------------------------------------
    # Display names
    # ----------------------------------------------------------

    display_names = display_names or {
        var: var.replace("_", " ")
        for var in categorical_vars
    }

    if metric_titles is None:
        metric_name_map = {
            metric: metric.replace("_", " ")
            for metric in metric_cols
        }
    else:
        metric_name_map = dict(zip(metric_cols, metric_titles))

    # ----------------------------------------------------------
    # Materials
    # ----------------------------------------------------------

    if selected_materials is None:
        materials = list(df_pipes[material_col].dropna().unique())
    else:
        materials = selected_materials

    # ----------------------------------------------------------
    # Find combinations with epsilon² >= threshold
    # ----------------------------------------------------------

    results_to_plot = []

    for metric in metric_cols:

        if metric not in epsilon_matrix:
            continue

        for cat in categorical_vars:

            for mat in materials:

                if (
                    cat not in epsilon_matrix[metric].index
                    or mat not in epsilon_matrix[metric].columns
                ):
                    continue

                eps = epsilon_matrix[metric].loc[cat, mat]

                if pd.notna(eps) and eps >= min_epsilon2:

                    results_to_plot.append({
                        "Material": mat,
                        "Variable": cat,
                        "Metric": metric,
                        "Epsilon2": eps
                    })

    # Sort from largest to smallest effect
    results_to_plot = sorted(
        results_to_plot,
        key=lambda x: x["Epsilon2"],
        reverse=True
    )

    # ----------------------------------------------------------
    # No results
    # ----------------------------------------------------------

    if len(results_to_plot) == 0:
        print(
            f"No results found with ε² ≥ {min_epsilon2:.3f}"
        )
        return None

    # ----------------------------------------------------------
    # Create subplot grid
    # ----------------------------------------------------------

    nplots = len(results_to_plot)
    ncols = min(ncols, nplots)
    nrows = math.ceil(nplots / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            figsize_per_plot[0] * ncols,
            figsize_per_plot[1] * nrows
        ),
        squeeze=False
    )

    axes = axes.flatten()

    # ----------------------------------------------------------
    # Plot each result
    # ----------------------------------------------------------

    for ax, result in zip(axes, results_to_plot):

        mat = result["Material"]
        cat = result["Variable"]
        metric = result["Metric"]
        eps = result["Epsilon2"]

        # Filter material
        sub = df_pipes[
            df_pipes[material_col] == mat
        ].copy()

        # Remove missing values
        sub = sub[
            sub[cat].notna()
            & sub[metric].notna()
        ]


        # Order of categories for this factor
        order = categories_order.get(cat, None)

        #Count number of pipes
        counts = sub.groupby(cat, observed=True).size()

        # Keep only categories that are actually present in the subset
        if order is not None:
            present_categories = set(sub[cat].dropna().unique())
            order = [x for x in order if x in present_categories]

        # Color according to material
        box_color = colors_materials.get(mat, "lightgrey")

        sns.boxplot(
            data=sub,
            x=cat,
            y=metric,
            order=order,
            ax=ax,
            showfliers=False,
            color=box_color
        )

        # Category names
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(
            order,
            fontsize=7
        )

        # Number of pipes underneath, with smaller font
        for i, category in enumerate(order):
            n_pipes = counts.get(category, 0)

            ax.text(
                i,
                -0.10,
                f"(pipes={n_pipes})",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6,
                rotation=45,
            )

        # Optional individual observations
        # sns.stripplot(
        #     data=sub,
        #     x=cat,
        #     y=metric,
        #     ax=ax,
        #     color="black",
        #     alpha=0.25,
        #     size=2
        # )

        # ------------------------------------------------------
        # Titles
        # ------------------------------------------------------

        variable_name = display_names.get(
            cat,
            cat.replace("_", " ")
        )

        metric_name = metric_name_map.get(
            metric,
            metric.replace("_", " ")
        )

        ax.set_title(
            f"{mat} | {variable_name} | {metric_name}",
            fontsize=8,
            fontweight="bold"
        )

        # ------------------------------------------------------
        # Axis labels
        # ------------------------------------------------------

        ax.set_xlabel(variable_name, fontsize=8)
        ax.set_ylabel(metric_name, fontsize=8)

        ax.tick_params(
            axis="x",
            rotation=45,
            labelsize=7
        )
        ax.tick_params(
            axis="y",
            labelsize=7
        )

        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

    # ----------------------------------------------------------
    # Remove unused axes
    # ----------------------------------------------------------

    for ax in axes[nplots:]:
        ax.remove()

    # fig.suptitle(
    #     f"Kruskal–Wallis results with ε² ≥ {min_epsilon2}",
    #     fontsize=13,
    #     fontweight="bold",
    #     y=1.01
    # )

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_categorical_analysis_combined(
    df_cctv_filtered,
    df_defect,
    categorical_vars,
    metric_cols,
    selected_materials=None,
    colors_materials=None,
    min_epsilon2=0.012,
    display_names=None,
    metric_titles=None,
    ncols=2,
    figsize_per_plot=(4.5, 4),
    analysis_kwargs=None,
    show=True,
):
    """Combine the Kruskal–Wallis heatmap and the boxplots"""

    analysis_params = {
        "id_col": "Pipe_ID",
        "material_col": "Material",
        "length_col": "Pipe_length",
        "categorical_vars": categorical_vars,
        "metric_cols": metric_cols,
        "display_names": display_names,
        "min_epsilon2": 0,
        "show": False,
    }
    analysis_params.update(analysis_kwargs or {})

    fig_heatmap, df_pipes, epsilon_matrix = analyze_categorical_kw(
        df_cctv_filtered=df_cctv_filtered,
        df_defect=df_defect,
        selected_materials=selected_materials,
        **analysis_params,
    )

    fig_boxplots = plot_boxplots_high_epsilon(
        df_pipes=df_pipes,
        epsilon_matrix=epsilon_matrix,
        categorical_vars=categorical_vars,
        metric_cols=metric_cols,
        selected_materials=selected_materials,
        colors_materials=colors_materials or {},
        min_epsilon2=min_epsilon2,
        display_names=display_names,
        metric_titles=metric_titles,
        ncols=ncols,
        figsize_per_plot=(3, 2.5),
        show=False,
    )

    if fig_boxplots is None:
        plt.close(fig_heatmap)
        return None, df_pipes, epsilon_matrix

    def figure_to_image(source_fig):
        source_fig.canvas.draw()
        return np.asarray(source_fig.canvas.buffer_rgba()).copy()

    heatmap_image = figure_to_image(fig_heatmap)
    boxplots_image = figure_to_image(fig_boxplots)

    heatmap_size = fig_heatmap.get_size_inches()
    boxplots_size = fig_boxplots.get_size_inches()

    output_width = heatmap_size[0]
    heatmap_height = heatmap_size[1]
    boxplots_height = output_width * 0.7 * boxplots_size[1] / boxplots_size[0]

    fig = plt.figure(figsize=(output_width, heatmap_height + boxplots_height))
    grid = fig.add_gridspec(
        2, 1,
        height_ratios=[heatmap_height, boxplots_height],
        hspace=0.16,
    )

    ax_top = fig.add_subplot(grid[0, 0])
    ax_bottom = fig.add_subplot(grid[1, 0])

    ax_top.imshow(heatmap_image)
    ax_top.axis("off")

    ax_boxplots = ax_bottom.inset_axes([0.15, 0, 0.70, 1])
    ax_boxplots.imshow(boxplots_image)
    ax_boxplots.axis("off")
    ax_bottom.axis("off")

    ax_top.text(
        0.5, 1.02, "(a)",
        transform=ax_top.transAxes,
        ha="center", va="bottom",
        fontweight="bold", clip_on=False,
    )
    ax_boxplots.text(
        0.5, 1.02, "(b)",
        transform=ax_boxplots.transAxes,
        ha="center", va="bottom",
        fontweight="bold", clip_on=False,
    )

    plt.close(fig_heatmap)
    plt.close(fig_boxplots)

    if show:
        plt.show()

    return fig, df_pipes, epsilon_matrix