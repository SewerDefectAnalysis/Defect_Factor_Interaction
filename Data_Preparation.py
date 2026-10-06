from typing import Mapping, Sequence,Tuple
from pathlib import Path
import pandas as pd

class DBReadError(Exception):
    """Custom exception for database read errors."""
    pass

class DataPreparationError(Exception):
    """Custom exception for data preparation errors."""
    pass

class FactorValidationError(Exception):
    """Custom exception for validating factor column names."""
    pass

def merge_df_pipes_hydraulic(
    df_pipes: pd.DataFrame,
    df_hydraulic: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge the pipes dataframe with the hydraulic dataframe using a left join
    on the column 'Pipe_ID'.

    Returns
    -------
    pd.DataFrame
        The merged dataframe.
    """
    if "Pipe_ID" not in df_pipes.columns or "Pipe_ID" not in df_hydraulic.columns:
        raise KeyError("Column 'Pipe_ID' not found")

    df_merged = df_pipes.merge(
        df_hydraulic,
        how="left",
        on="Pipe_ID"
    )

    return df_merged

def filter_by_material(
    df: pd.DataFrame,
    selected_materials: Sequence[str],
    material_column: str = "Material"
) -> pd.DataFrame:
    """
    Filter any DataFrame by a list of selected materials.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a material column.
    selected_materials : list-like of str
        Materials to filter by.
    material_column : str, default "Material"
        Name of the column containing material codes.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.

    Raises
    ------
    DataPreparationError
        If material column is missing or no selected materials are present.
    """
    # 1. Validate that the column exists
    if material_column not in df.columns:
        raise DataPreparationError(
            f"Column '{material_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # 2. Materials available in dataset
    materials_in_df = df[material_column].dropna().unique().tolist()

    # 3. Check for missing materials
    missing = [m for m in selected_materials if m not in materials_in_df]
    if missing:
        raise DataPreparationError(
            f"Selected materials not found: {missing}. "
            f"Available materials: {materials_in_df}"
        )

    # 4. Filter
    df_filtered = df[df[material_column].isin(selected_materials)].copy()

    if df_filtered.empty:
        raise DataPreparationError(
            "Filtering resulted in an empty DataFrame."
        )

    return df_filtered

def merge_cctv_with_material(
    df_pipes: pd.DataFrame,
    df_cctv: pd.DataFrame,
    comp_col: str = "Pipe_ID",
    material_col: str = "Material",
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge material to the CCTV dataframe from the pipes dataframe.

    Parameters
    ----------
    df_pipes: pd.DataFrame
        Pipes dataframe
    df_cctv : pd.DataFrame
        CCTV dataframe containing (Pipe_ID, Material).
    comp_col : str, default "Pipe_ID"
        Name of the join key column.
    age_col : str, default "Material"
        Name of the age column in df_cctv.
    how : str, default "inner"
        Type of merge to perform (passed to pd.merge).

    Returns
    -------
    pd.DataFrame
        Pipes dataframe with an additional Material column.
    """
    df_pipes = df_pipes.copy()
    df_cctv = df_cctv.copy()

    df_merged = pd.merge(df_pipes, df_cctv, on=comp_col, how=how)

    return df_merged



def validate_factors_in_dataframe(
    df: pd.DataFrame,
    selected_factors: Sequence[str],
    strict: bool = False
) -> list[str]:
    """
    Validate that selected factor names exist in the DataFrame columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing pipe attributes.
    selected_factors : list-like of str
        List of factors to validate.
    strict : bool, default False
        - If True: raise an error if any factor is missing.
        - If False: missing factors are ignored with a warning.

    Returns
    -------
    list of str
        The subset of selected factors that exist in the DataFrame.

    Raises
    ------
    FactorValidationError
        If strict=True and one or more factors are missing.
    """

    available_factors = df.columns.tolist()

    missing = [f for f in selected_factors if f not in available_factors]

    if missing:
        message = (
            f"The following factors are NOT present in the dataframe: {missing}. "
            f"Available factors: {available_factors}"
        )

        if strict:
            raise FactorValidationError(message)
        else:
            print("⚠️ Warning:", message)
            print("These missing factors will be ignored.")

    # Return only the factors that exist
    valid_factors = [f for f in selected_factors if f in available_factors]

    return valid_factors


def auto_classify_factors(
    df: pd.DataFrame,
    factors: Sequence[str],
) -> Tuple[list[str], list[str]]:
    """
    Automatically classify selected factors as numeric or categorical
    based on the dataframe dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the factors.
    factors : list-like of str
        List of factor names to classify.

    Returns
    -------
    factors_num : list[str]
        Factors detected as numeric.
    factors_cat : list[str]
        Factors detected as categorical.
    """
    # Automatically classification of numeric and categorical factors
    numeric_columns = df.select_dtypes(include=['number']).columns
    categorical_columns = df.select_dtypes(exclude=['number']).columns

    factors_num = [f for f in factors if f in numeric_columns]
    factors_cat = [f for f in factors if f in categorical_columns]

    return factors_num, factors_cat

def merge_defects_with_pipes(
    df_defect: pd.DataFrame,
    df_pipes_filtered: pd.DataFrame,
    pipe_columns: Sequence[str],
    drop_defect_columns: Sequence[str] = ("Material", "Length"),
    on: str = "Pipe_ID",
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge the defects dataframe with pipe information.

    Steps
    -----
    1) Optionally drop repeated/duplicated columns from the defects dataframe
       (e.g. 'Material', 'Length').
    2) Select a subset of columns from the pipes dataframe.
    3) Merge both dataframes on the pipe identifier (Pipe_ID by default).

    Parameters
    ----------
    df_defect : pd.DataFrame
        Defects dataframe.
    df_pipes_filtered : pd.DataFrame
        Pipes dataframe, already filtered as needed.
    pipe_columns : sequence of str
        List of columns to keep from df_pipes_filtered for the merge.
    drop_defect_columns : sequence of str, default ("Material", "Length")
        Columns to drop from df_defect before merging (to avoid duplicates).
    on : str, default "Pipe_ID"
        Name of the join key column.
    how : str, default "inner"
        Type of merge to perform (passed to pd.merge).

    Returns
    -------
    pd.DataFrame
        Merged dataframe with defects and pipe attributes.
    """
    df_def = df_defect.copy()
    df_pipes = df_pipes_filtered.copy()

    # 1) Drop duplicated columns in the defects dataframe
    cols_to_drop = [c for c in drop_defect_columns if c in df_def.columns]
    if cols_to_drop:
        df_def = df_def.drop(columns=cols_to_drop)

    # 2) Select only the desired columns from pipes
    df_pipes_sub = df_pipes[list(pipe_columns)]

    # 3) Merge defects with pipe information
    merged = pd.merge(df_pipes_sub, df_def, on=on, how=how)

    return merged


def merge_pipes_with_age(
    df: pd.DataFrame,
    df_cctv_filtered: pd.DataFrame,
    comp_col: str = "Pipe_ID",
    age_col: str = "Age_CCTV",
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge age at inspection from the CCTV dataframe into the pipes dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe, already filtered as needed.
    df_cctv_filtered : pd.DataFrame
        CCTV dataframe containing (Pipe_ID, Age_cctv).
    comp_col : str, default "Pipe_ID"
        Name of the join key column.
    age_col : str, default "Age_cctv"
        Name of the age column in df_cctv_filtered.
    how : str, default "inner"
        Type of merge to perform (passed to pd.merge).

    Returns
    -------
    pd.DataFrame
        Pipes dataframe with an additional Age_CCTV column.
    """
    df = df.copy()
    df_cctv = df_cctv_filtered[[comp_col, age_col]].copy()

    df_merged = pd.merge(df_cctv, df, on=comp_col, how=how)

    return df_merged


def add_total_defects_per_pipe(df_pipes: pd.DataFrame,
                               df_defects: pd.DataFrame,
                               pipe_id_col: str = 'Pipe_ID',
                               merge_key: str = 'Pipe_ID',
                               out_col: str = 'Total_defects') -> pd.DataFrame:
    """
    Calculate the total number of defects per pipe and merge the result
    into the pipes dataframe.

    Parameters
    ----------
    df_pipes : pd.DataFrame
        DataFrame containing pipe information.
    df_defects : pd.DataFrame
        DataFrame containing defect records. Must include the pipe ID column.
    pipe_id_col : str, default 'Pipe_ID'
        Column in df_defects identifying the pipe.
    merge_key : str, default 'Pipe_ID'
        Column in df_pipes used to merge defect counts.
    out_col : str, default 'Total_defects'
        Name of the output column that will store total defects per pipe.

    Returns
    -------
    pd.DataFrame
        Copy of df_pipes including a new column with defect counts.
    """

    df_pipes = df_pipes.copy()

    # --- Count defects per pipe ---
    defect_counts = df_defects[pipe_id_col].value_counts().reset_index()
    defect_counts.columns = [merge_key, out_col]

    # --- Merge into pipes dataframe ---
    df_pipes = df_pipes.merge(defect_counts, on=merge_key, how='left')

    # --- Fill missing values with 0 (pipes with no defects) ---
    df_pipes[out_col] = df_pipes[out_col].fillna(0).astype(int)

    return df_pipes

def merge_cctv_defects(
        df_cctv: pd.DataFrame,
        df_pipes_filtered: pd.DataFrame,
        df_defects: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge filtered CCTV and pipe data into the defects dataframe.

    Parameters:
    - df_cctv: DataFrame with CCTV inspections
    - df_pipes_filtered: DataFrame with filtered pipe information
    - df_defects: DataFrame with defects to be enriched

    Returns:
    - df_defects merged with CCTV and pipe data
    """
    # Merge CCTV with filtered pipes
    df_cctv_filtered = pd.merge(df_cctv, df_pipes_filtered, on='Pipe_ID', how='inner')

    # Merge defects with the enriched CCTV data
    df_defects_filtered = pd.merge(df_defects, df_cctv_filtered, on='Pipe_ID', how='inner')

    return df_cctv_filtered, df_defects_filtered


def prepare_analysis_datasets(
    df_defects,
    df_pipes_filtered,
    df_cctv_filtered,
    pipe_columns_for_defects,
    pipe_id_col="Pipe_ID",
    age_col="Age_CCTV",
    total_defects_col="Total_defects",
):
    """
    Prepare the main datasets used in the defect-level analysis.

    This function:
    1. Adds selected pipe attributes to each defect.
    2. Adds CCTV pipe age to the pipe dataset.
    3. Adds CCTV pipe age to the defect dataset.
    4. Adds the total number of defects per pipe to the pipe dataset.
    """

    df_defects_prepared = merge_defects_with_pipes(
        df_defect=df_defects,
        df_pipes_filtered=df_pipes_filtered,
        pipe_columns=pipe_columns_for_defects,
        drop_defect_columns=("Material", "Length"),
        on=pipe_id_col,
        how="inner",
    )

    df_pipes_prepared = merge_pipes_with_age(
        df=df_pipes_filtered,
        df_cctv_filtered=df_cctv_filtered,
        comp_col=pipe_id_col,
        age_col=age_col,
        how="inner",
    )

    df_defects_prepared = merge_pipes_with_age(
        df=df_defects_prepared,
        df_cctv_filtered=df_cctv_filtered,
        comp_col=pipe_id_col,
        age_col=age_col,
        how="inner",
    )

    df_pipes_prepared = add_total_defects_per_pipe(
        df_pipes=df_pipes_prepared,
        df_defects=df_defects_prepared,
        pipe_id_col=pipe_id_col,
        merge_key=pipe_id_col,
        out_col=total_defects_col,
    )

    return df_defects_prepared, df_pipes_prepared

def standardize(df_pipes):

    # Standardize soil type names
    df_pipes["Soil_type"] = df_pipes["Soil_type"].replace({
        "Boulders to massive": "Boulders"
    })

    return df_pipes


DEFECT_CODE_MAP = {
    "B": "Blocked pipe",
    "CC": "Cracking Circumferential",
    "CL": "Cracking Longitudinal",
    "CM": "Cracking Multiple",
    "DE": "Debris Silty",
    "DF": "Deformed Pipe",
    "DG": "Debris Greasy",
    "DP": "Dipped Pipe",
    "ED": "Encrustation Deposits",
    "EX": "Exfiltration",
    "IP": "Infiltration Present",
    "JD": "Joint Displaced",
    "JF": "Joint Faulty",
    "JO": "Joint Open",
    "LF": "Lateral Sealing Faulty",
    "LP": "Lateral Protruding",
    "LX": "Lateral Problem",
    "MHJ": "Manhole Joint Faulty",
    "O": "Obstruction",
    "PB": "Pipe Broken",
    "PF": "Deformed Plastic Pipe",
    "PH": "Pipe Holed",
    "PL": "Protective Lining Defective",
    "RI": "Root Intrusion",
    "S": "Surface Damage",
    "SV": "Soil Visible",
    "TM": "Tomo",
}

def add_defect_code_full(df):
    """
    Add a column 'Defect_code_full' based on 'Defect_code'.
    """
    df = df.copy()
    df["Defect_code_full"] = df["Defect_code"].map(DEFECT_CODE_MAP)
    return df