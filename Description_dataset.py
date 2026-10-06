from typing import Mapping, Sequence,Tuple
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from Save_figure import save_figure_pdf
import math

def plot_boxplots_grid(
    df: pd.DataFrame,
    factors: list,
    display_names: dict,
    group_col: str,
    selected_groups: list | None = None,
    colors: dict | None = None,
    n_cols: int = 4,
    units: dict | None = None,

):
    """
    Create a grid of boxplots for numeric variables grouped by a categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    factors : list
        List of numeric variables to plot.
    display_names : dict
        Dictionary mapping factor names to nicer axis labels.
    group_col : str
        Column used for grouping on the x-axis (e.g., MATERIAL).
    selected_groups : list, optional
        Order of categories in the x-axis.
    colors : dict, optional
        Dictionary mapping categories to colors.
    n_cols : int, default 3
        Number of columns in the subplot grid.
    """

    # --- Validation --- #
    if df is None or df.empty:
        raise ValueError("The input DataFrame is empty.")

    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in DataFrame.")

    for col in factors:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # --- Grid layout --- #
    n_vars = len(factors)
    n_rows = math.ceil(n_vars / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows)
    )

    axes = axes.flatten()

    # --- Plot boxplots --- #
    for i, factor in enumerate(factors):

        sns.boxplot(
            data=df,
            x=group_col,
            y=factor,
            hue=group_col,
            order=selected_groups,
            palette=colors,
            width=0.75,
            ax=axes[i],
            legend=False,
        )

        label = display_names.get(factor, factor)

        if units and factor in units:
            label = f"{label} ({units[factor]})"

        axes[i].set_ylabel(label,fontsize=13.5)
        axes[i].set_xlabel("",fontsize=13)

        # Axis tick labels
        axes[i].tick_params(
            axis="x",
            labelsize=12.5,
            rotation=0
        )

        axes[i].tick_params(
            axis="y",
            labelsize=12.5
        )

        # Title
        axes[i].set_title(
            f"Distribution of {label}",
            fontsize=13,
            fontweight="bold"
        )

    # --- Remove unused axes --- #
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.subplots_adjust(
    hspace=0.65,
)
    plt.tight_layout()
    plt.show()
    return fig



def create_range_table(df, factors, display_names=None):
    """
    Create a table with the range [min–max] for numeric factors.
    """

    rows = []

    for factor in factors:
        min_val = df[factor].min()
        max_val = df[factor].max()

        range_str = f"[{min_val:.3f}–{max_val:.3f}]"

        name = display_names.get(factor, factor) if display_names else factor

        rows.append({
            "Factor": name,
            "Range": range_str
        })

    table = pd.DataFrame(rows)

    return table

import pandas as pd


def create_categorical_summary_table(
    df: pd.DataFrame,
    categorical_vars: list,
    material_col: str = "Material",
    material_order: list | None = None,
    display_names: dict | None = None,
    categories_order: dict | None = None,
    category_labels: dict | None = None,
    decimals: int = 1,
):
    """
    Create a summary table showing the percentage distribution of categorical
    variables within each material.

    Each cell contains the categories and percentages separated by line breaks.
    """

    if df is None or df.empty:
        raise ValueError("The input DataFrame is empty.")

    if material_col not in df.columns:
        raise ValueError(
            f"Column '{material_col}' was not found in the DataFrame."
        )

    missing_variables = [
        var for var in categorical_vars
        if var not in df.columns
    ]

    if missing_variables:
        raise ValueError(
            f"The following categorical variables were not found: "
            f"{missing_variables}"
        )

    if material_order is None:
        material_order = (
            df[material_col]
            .dropna()
            .unique()
            .tolist()
        )

    display_names = display_names or {}
    categories_order = categories_order or {}
    category_labels = category_labels or {}

    rows = []

    for variable in categorical_vars:

        row = {
            "Variable": display_names.get(variable, variable)
        }

        for material in material_order:

            subset = df.loc[
                df[material_col] == material,
                variable
            ]

            valid_values = subset.dropna()

            if valid_values.empty:
                row[material] = "No data"
                continue

            percentages = (
                valid_values
                .value_counts(normalize=True)
                .mul(100)
            )

            # Use the predefined category order when available
            variable_order = categories_order.get(variable)

            if variable_order is not None:
                categories = [
                    category
                    for category in variable_order
                    if category in percentages.index
                ]
            else:
                categories = percentages.index.tolist()

            cell_lines = []

            for category in categories:

                # Convert values such as 0.0 and 1.0 to No and Yes
                label = category_labels.get(
                    variable,
                    {}
                ).get(
                    category,
                    str(category)
                )

                percentage = percentages.loc[category]

                cell_lines.append(
                    f"{label}: {percentage:.{decimals}f}%"
                )

            # HTML line break for notebook display
            row[material] ="\n".join(cell_lines)

        rows.append(row)

    return pd.DataFrame(rows)



def corr_matrix(df, num_vars, display_names,method):
    """
    Plot Spearman correlation matrix for numerical variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the numerical variables.
    num_vars : list
        List of column names to include in the correlation matrix.
    display_names : dict
        Dictionary mapping original variable names to display names for labels.
    methods : String
        Select the method to calculate the correlations. .


    Returns
    -------
    matplotlib.figure.Figure
        The generated correlation heatmap figure.
    """

    # Compute correlation matrix
    corr_matrix = df[num_vars].corr(method=method)

    # Rename variables for display
    corr_renamed = corr_matrix.rename(index=display_names, columns=display_names)

    return corr_renamed

# --- Helper: entropy for numeric/categorical variables ---
def _entropy_mixed(x, kind, n_bins=10):
    """Entropy in nats for numeric/categorical/binary data."""
    x = np.asarray(x)
    if kind == "num":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(x, qs))
        if edges.size < 3:
            return 0.0
        idx = np.clip(np.digitize(x, edges[1:-1], right=True), 0, edges.size - 2)
        counts = np.bincount(idx, minlength=edges.size - 1).astype(float)
    else:
        _, counts = np.unique(x, return_counts=True)
        counts = counts.astype(float)

    p = counts / counts.sum() if counts.sum() > 0 else counts
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# --- Main NMI function ---
def mutual_info_categorical_vs_all_mixed(
    df: pd.DataFrame,
    numerical: list,
    categorical: list,
    binary: list | None = None,
    display_names: dict | None = None,
    *,
    n_bins: int = 10,
    n_neighbors: int = 3,
    random_state: int = 42,
):
    """
    Compute NMI between categorical variables (rows)
    and numerical/categorical/binary variables (columns).
    NaNs in categorical variables are replaced with "Unknown"
    ONLY inside a working copy (df_nmi).
    """

    if binary is None:
        binary = []

    all_vars = numerical + categorical + binary
    mi_matrix = pd.DataFrame(index=categorical, columns=all_vars, dtype=float)

    # Variable type map
    kind_map = {
        c: ("num" if c in numerical else "cat" if c in categorical else "bin")
        for c in all_vars
    }

    scaler = StandardScaler()

    # --- Create a working copy for NMI only ---
    df_nmi = df.copy()
    df_nmi[categorical] = df_nmi[categorical].fillna("Unknown")

    # --- Precompute entropies ---
    entropies = {}
    for c in all_vars:
        col = df_nmi[c].dropna()
        if col.empty:
            entropies[c] = 0.0
            continue

        kind = kind_map[c]
        entropies[c] = _entropy_mixed(col, kind, n_bins=n_bins)

    # --- Compute MI for each pair ---
    for cat in categorical:
        y_kind = kind_map[cat]

        for v in all_vars:
            if v == cat:
                mi_matrix.loc[cat, v] = 1.0
                continue

            subset = df_nmi[[cat, v]].dropna()
            if subset.shape[0] < max(5, n_neighbors + 1):
                mi_matrix.loc[cat, v] = np.nan
                continue

            xa = subset[v]
            ya = subset[cat]

            x_kind = kind_map[v]

            # --- Encode variables ---
            # Encode X
            if x_kind == "num":
                x_enc = scaler.fit_transform(xa.to_numpy().reshape(-1, 1))
            else:
                x_enc, _ = pd.factorize(xa.fillna("Unknown"))
                x_enc = x_enc.reshape(-1, 1)

            # Encode Y (categorical)
            if y_kind == "num":
                y_enc = scaler.fit_transform(ya.to_numpy().reshape(-1, 1)).ravel()
            else:
                y_enc, _ = pd.factorize(ya.fillna("Unknown"))

            # --- Compute MI ---
            try:
                if (x_kind == "num") and (y_kind == "num"):
                    mi = float(
                        mutual_info_regression(
                            x_enc, y_enc, n_neighbors=n_neighbors, random_state=random_state
                        )[0]
                    )
                else:
                    y_labels = y_enc
                    mi = float(
                        mutual_info_classif(
                            x_enc, y_labels, random_state=random_state
                        )[0]
                    )
            except Exception:
                mi_matrix.loc[cat, v] = np.nan
                continue

            # --- Compute NMI (symmetric version) ---
            Hx = entropies.get(v, 0.0)
            Hy = entropies.get(cat, 0.0)

            if (Hx + Hy) == 0:
                mi = 0.0
            else:
                mi = 2 * mi / (Hx + Hy)
                mi = float(np.clip(mi, 0, 1))

            mi_matrix.loc[cat, v] = mi

    # --- Apply display names ---
    if display_names:
        mi_matrix = mi_matrix.rename(
            index={k: display_names.get(k, k) for k in mi_matrix.index},
            columns={k: display_names.get(k, k) for k in mi_matrix.columns},
        )

    # --- Final diagonal = 1 ---
    common = [v for v in mi_matrix.index if v in mi_matrix.columns]
    for v in common:
        mi_matrix.loc[v, v] = 1.0

    return mi_matrix


def plot_corr_and_nmi_panels(
    corr_matrix: pd.DataFrame,
    mi_matrix: pd.DataFrame,
    figsize: Tuple[float, float] = (7, 9),
        tick_size: int = 9,
        label_size: int = 9,
        cbar_tick_size: int = 8,
        cbar_label_size: int = 9,
        cbar_width_scale: float = 0.5,
) -> plt.Figure:
    fig = plt.figure(figsize=figsize)

    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.3, 0.08, 0.45],
        width_ratios=[0.90, 0.05],
        hspace=0.35,
        wspace=0.05,
    )

    ax_a, cax_a = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_b, cax_b = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

    # -------- PLOT (a): correlation --------
    hm_a = sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        annot=False,
        square=True,
        cbar_ax=cax_a,
        ax=ax_a,
    )

    ax_a.set_aspect("equal")
    ax_a.tick_params(axis="x", rotation=90, labelsize=tick_size)
    ax_a.tick_params(axis="y", rotation=0, labelsize=tick_size)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("")
    ax_a.text(0.5, 1.05, "(a)", transform=ax_a.transAxes,
              fontsize=label_size, ha="center", va="bottom", fontweight="bold")

    # -------- PLOT (b): NMI --------
    hm_b = sns.heatmap(
        mi_matrix,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        annot=False,
        cbar_ax=cax_b,
        ax=ax_b,
    )


    posb = ax_b.get_position()
    ax_b.set_position([posb.x0, posb.y0 - 0.05, posb.width, posb.height])

    ax_b.set_aspect(0.22 * mi_matrix.shape[1] / mi_matrix.shape[0])
    ax_b.tick_params(axis="x", rotation=90, labelsize=tick_size)
    ax_b.tick_params(axis="y", rotation=0, labelsize=tick_size)
    ax_b.set_xlabel("")
    ax_b.set_ylabel("")
    ax_b.text(0.5, 1.05, "(b)", transform=ax_b.transAxes,
              fontsize=label_size, ha="center", va="bottom", fontweight="bold")

    # -------- Colorbars --------
    # Upper Colorbar
    pos_cb_a = cax_a.get_position()
    cax_a.set_position([
        pos_cb_a.x0,
        pos_cb_a.y0,
        pos_cb_a.width * cbar_width_scale,
        pos_cb_a.height
    ])
    cbar_a = hm_a.collections[0].colorbar
    if cbar_a is not None:
        cbar_a.ax.tick_params(labelsize=cbar_tick_size)
        cbar_a.set_label(
            "Spearman correlation coefficient",
            rotation=90, fontsize=cbar_label_size, labelpad=20
        )

    # Bottom Colorbar
    pos_cb_b = cax_b.get_position()
    pos_b = ax_b.get_position()
    cax_b.set_position([
        pos_cb_b.x0 + 0.01,
        pos_b.y0,
        pos_cb_b.width * cbar_width_scale,
        pos_b.height
    ])
    cbar_b = hm_b.collections[0].colorbar
    if cbar_b is not None:
        cbar_b.ax.tick_params(labelsize=cbar_tick_size)
        cbar_b.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        cbar_b.set_label(
            "Normalized Mutual Information index",
            rotation=90, fontsize=cbar_label_size, labelpad=20
        )
    plt.show()
    return fig

