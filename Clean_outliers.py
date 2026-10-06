from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def clean_outliers_by_material_isoforest_cell(
    df: pd.DataFrame,
    material_col: str = "Material",
    numeric_cols: Optional[Sequence[str]] = None,
    iso_contamination: float = 0.05,    # expected outlier fraction per material block
    cell_mz_thresh: float = 3.5,        # modified Z-score threshold for per-cell masking
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect outliers within each material group using a *multivariate* IsolationForest model,
    and mask only suspicious cells (set to NaN). No row deletions are performed.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing at least the material column and the numeric feature columns.
    material_col : str, default="Material"
        Column that defines groups (one IsolationForest model is fit per material).
    numeric_cols : Sequence[str] or None, default=None
        Numeric feature columns to use. If None, inferred from numeric dtypes (excluding `material_col`).
    iso_contamination : float, default=0.05
        Proportion of expected outliers for IsolationForest (per material block).
    cell_mz_thresh : float, default=3.5
        Absolute modified Z-score threshold to decide which cells to mask.
    random_state : int, default=42
        Random seed for IsolationForest.

    Returns
    -------
    df_clean : pd.DataFrame
        Same shape as input; only certain numeric cells may be set to NaN.
    summary : pd.DataFrame
        Per-material summary with counts of rows in and cells masked.

    Notes
    -----
    - Workflow per material:
        1) Fit IsolationForest on median-imputed numeric features.
        2) For rows flagged as outliers, compute per-feature modified Z-score relative
           to the *inlier* subset (median & MAD). Mask cells with |mz| > threshold.
    - Features with MAD == 0 among inliers are skipped (no masking for that feature).
    """
    df = df.copy()

    # Basic check: material column must exist
    if material_col not in df.columns:
        raise ValueError(
            f"Column '{material_col}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    # Infer numeric columns if not provided (excluding material_col if numeric)
    if numeric_cols is None:
        inferred = df.select_dtypes(include=[np.number]).columns.tolist()
        if material_col in inferred:
            inferred.remove(material_col)
        numeric_cols = inferred

    # Keep only existing numeric columns
    cols_present = [c for c in numeric_cols if c in df.columns]
    if not cols_present:
        # Nothing to clean; return original df and empty summary
        empty_summary = pd.DataFrame(columns=["material", "rows_in", "cells_masked"])
        return df, empty_summary

    mats = df[material_col].dropna().unique()
    summary_rows = []

    for mat in mats:
        block_idx = df[material_col] == mat
        block = df.loc[block_idx].copy()
        n0 = len(block)
        cells_masked = 0

        if n0 == 0:
            summary_rows.append({"material": mat, "rows_in": 0, "cells_masked": 0})
            continue

        # 1) Fit IsolationForest on this material block
        X = block[cols_present]
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("iso", IsolationForest(
                contamination=iso_contamination,
                random_state=random_state
            )),
        ])
        preds = pipe.fit_predict(X)          # +1 inliers, -1 outliers
        outlier_rows = (preds == -1)

        # 2) Compute robust center/scale using only inliers
        inliers = block.loc[~outlier_rows, cols_present]
        if len(inliers) > 0:
            med = inliers.median(numeric_only=True)
            mad = (inliers - med).abs().median(numeric_only=True)
            safe_mad = mad.replace(0.0, np.nan)

            sub = block.loc[outlier_rows, cols_present]
            # Modified Z-score: mz = 0.6745 * (x - med) / MAD
            mz = 0.6745 * (sub - med) / safe_mad

            mask_cells = mz.abs() > cell_mz_thresh
            cells_masked = int(mask_cells.fillna(False).to_numpy().sum())

            # Apply mask only where condition holds True
            block.loc[mask_cells.index, cols_present] = (
                block.loc[mask_cells.index, cols_present].mask(mask_cells)
            )

            # Write updated cells back to master df
            df.loc[block.index, cols_present] = block[cols_present]

        # If no inliers, we skip masking and just keep data as-is
        summary_rows.append({
            "material": mat,
            "rows_in": int(n0),
            "cells_masked": int(cells_masked),
        })

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values("material")
        .reset_index(drop=True)
    )
    return df, summary_df
