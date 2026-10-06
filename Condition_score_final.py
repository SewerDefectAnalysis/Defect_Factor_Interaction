from Config import _service_scores, _structural_scores, VALID_QUANTIFICATIONS
import numpy as np
import pandas as pd


# ============================================================
# Score code configuration
# ============================================================

NULL_TEXT_VALUES = {
    "": pd.NA,
    "NAN": pd.NA,
    "NONE": pd.NA,
    "<NA>": pd.NA,
}

# ============================================================
# Defect score code preparation
# ============================================================
def validate_quantification_values(
    defects,
    quantification_column="Quantification",
    valid_quantifications=VALID_QUANTIFICATIONS,
):
    """
    Check that all non-missing quantification values are valid.

    Raises
    ------
    ValueError
        If invalid quantification values are found.
    """

    quantification = clean_text_column(
        defects[quantification_column]
    )

    invalid_mask = (
        quantification.notna()
        & ~quantification.isin(valid_quantifications)
    )

    if invalid_mask.any():
        invalid_values = (
            quantification.loc[invalid_mask]
            .value_counts()
            .rename_axis("Quantification")
            .reset_index(name="Count")
            .to_dict(orient="records")
        )

        raise ValueError(
            "Invalid Quantification values were found: "
            f"{invalid_values}"
        )

def size_validation(
    df_defects: pd.DataFrame,
    structural_scores: dict,
    service_scores: dict,
    size_suffixes: set[str] = VALID_QUANTIFICATIONS,
) -> pd.DataFrame:
    """
    Fill missing defect quantifications using the most frequent
    Quantification observed for defect codes that require a size
    classification according to the structural and service score dictionaries.

    Parameters
    ----------
    df_defects : pd.DataFrame
        DataFrame containing defect information.

    structural_scores : dict
        Dictionary containing structural defect scores.

    service_scores : dict
        Dictionary containing service defect scores.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing Quantification values filled using
        the most frequent value for each applicable defect code.
    """

    df_defects = df_defects.copy()

    all_score_codes = set(structural_scores) | set(service_scores)

    existing_defect_codes = set(
        df_defects["Defect_code"]
        .dropna()
        .unique()
    )

    size_defect_codes = {
        score_code[:-1]
        for score_code in all_score_codes
        if (
            len(score_code) > 1
            and score_code[-1] in size_suffixes

            # The complete score code must NOT already be a defect code
            and score_code not in existing_defect_codes

            # The part before L/M/S must be a real defect code
            and score_code[:-1] in existing_defect_codes
        )
    }

    df_defects["Quantification"] = (
        df_defects["Quantification"]
        .replace(r"^\s*$", np.nan, regex=True)
    )

    most_frequent_size = (
        df_defects.loc[
            df_defects["Defect_code"].isin(size_defect_codes)
            & df_defects["Quantification"].notna()
        ]
        .groupby("Defect_code")["Quantification"]
        .agg(
            lambda x: (
                x.mode().iloc[0]
                if not x.mode().empty
                else np.nan
            )
        )
    )


    mask_missing = (
        df_defects["Quantification"].isna()
        & df_defects["Defect_code"].isin(size_defect_codes)
    )

    print(
        f"Missing quantifications to correct: "
        f"{mask_missing.sum()}"
    )

    df_defects.loc[
        mask_missing,
        "Quantification"
    ] = (
        df_defects.loc[
            mask_missing,
            "Defect_code"
        ]
        .map(most_frequent_size)
    )

    mask_still_missing = (
            df_defects["Quantification"].isna()
            & df_defects["Defect_code"].isin(size_defect_codes)
    )

    if mask_still_missing.any():
        missing_by_defect = (
            df_defects.loc[
                mask_still_missing,
                "Defect_code"
            ]
            .value_counts()
            .rename_axis("Defect_code")
            .reset_index(name="Count")
            .to_dict(orient="records")
        )

        raise ValueError(
            "Quantification could not be determined for "
            "defects that require a size classification: "
            f"{missing_by_defect}"
        )

    return df_defects

def clean_text_column(series):
    """
    Clean a text column by removing surrounding spaces, converting
    values to uppercase, and replacing common null representations
    with pandas missing values.
    """
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace(NULL_TEXT_VALUES)
    )

def validate_pipe_lengths(
    defects,
    pipe_id_column="Pipe_ID",
    pipe_length_column="Pipe_length",
):
    """
    Check that pipe lengths required for the condition score
    are available and greater than zero.

    Raises
    ------
    ValueError
        If missing, zero, or negative pipe lengths are found.
    """

    pipe_length = pd.to_numeric(
        defects[pipe_length_column],
        errors="coerce",
    )

    invalid_mask = (
        pipe_length.isna()
        | (pipe_length <= 0)
    )

    if invalid_mask.any():

        invalid_pipes = (
            defects.loc[
                invalid_mask,
                [pipe_id_column, pipe_length_column]
            ]
            .drop_duplicates()
        )

        missing_count = (
            pipe_length.loc[invalid_mask]
            .isna()
            .sum()
        )

        non_positive_count = (
            (pipe_length.loc[invalid_mask] <= 0)
            .sum()
        )

        invalid_values = (
            invalid_pipes
            .to_dict(orient="records")
        )

        raise ValueError(
            "Invalid Pipe_length values were found. "
            f"Missing: {missing_count}; "
            f"Zero or negative: {non_positive_count}. "
            f"Affected pipes: {invalid_values}"
        )

def create_score_code(
    defect_code,
    quantification,
    structural_scores=_structural_scores,
    service_scores=_service_scores,
):
    """
    Create the score lookup code by combining the defect code
    and its quantification.

    If the combined code does not exist in either score dictionary,
    but the defect code alone does exist, use the defect code alone.

    Examples
    --------
    CC + L  -> CCL
    DP + M  -> DPM
    B  + NA -> B
    PX + NA -> PX
    TM + L  -> TM
    """

    if pd.isna(defect_code):
        return pd.NA

    defect_code = str(defect_code).strip().upper()

    # If quantification is missing, use defect code only
    if pd.isna(quantification):
        return defect_code

    quantification = str(quantification).strip().upper()

    # If quantification is invalid, use defect code only
    if quantification not in VALID_QUANTIFICATIONS:
        return defect_code

    # Create candidate code with quantification
    score_code_with_size = defect_code + quantification

    # All valid score codes
    valid_score_codes = (
        set(structural_scores)
        | set(service_scores)
    )

    # Use code with quantification if it exists
    if score_code_with_size in valid_score_codes:
        return score_code_with_size

    # Otherwise, use defect code alone if it exists
    if defect_code in valid_score_codes:
        return defect_code

    # If neither exists, return the combined code so that
    # the unknown-score validation can detect it later
    return score_code_with_size


def add_defect_scores(
    defects,
    defect_code_column="Defect_code",
    quantification_column="Quantification",
):
    """
    Clean defect codes, create the score lookup code, and assign the
    structural and service scores.

    Parameters
    ----------
    defects : pandas.DataFrame
        Defect-level dataset.

    Returns
    -------
    pandas.DataFrame
        Copy of the defect dataset containing Score_code,
    """
    result = defects.copy()

    result["Defect_code_clean"] = clean_text_column(
        result[defect_code_column]
    )

    result["Quantification_clean"] = clean_text_column(
        result[quantification_column]
    )

    result["Score_code"] = [
        create_score_code(defect_code, quantification)
        for defect_code, quantification in zip(
            result["Defect_code_clean"],
            result["Quantification_clean"],
        )
    ]

    result["Structural_score"] = result["Score_code"].map(
        _structural_scores
    )

    result["Service_score"] = result["Score_code"].map(
        _service_scores
    )

    return result

def find_unknown_score_codes(
    defects
):
    """
    Return score codes that are not defined in either score dictionary.
    """
    valid_codes = set(_structural_scores) | set(_service_scores)

    unknown_mask = (
        defects["Score_code"].notna()
        & ~defects["Score_code"].isin(valid_codes)
    )

    return (
        defects.loc[unknown_mask, "Score_code"]
        .value_counts()
        .rename_axis("Score_code")
        .reset_index(name="Count")
    )

def add_defect_positions(
    defects,
    pipes,
    pipe_id_column="Pipe_ID",
    pipe_length_column="Pipe_length",
    longitudinal_distance_column="Longitudinal_distance",
    normalized_defect_length_column="Defect_length",
):
    """
    Add pipe length and calculate the start, end, and actual length
    of each defect in metres.

    This function assumes that Longitudinal_distance is already
    expressed in metres and that Defect_length is normalized relative
    to the pipe length.
    """
    result = defects.copy()

    # Remove existing pipe length to avoid _x / _y after merge
    if pipe_length_column in result.columns:
        result = result.drop(columns=pipe_length_column)

    pipe_lengths = (
        pipes[
            [
                pipe_id_column,
                pipe_length_column,
            ]
        ]
        .drop_duplicates(subset=pipe_id_column)
    )

    result = result.merge(
        pipe_lengths,
        on=pipe_id_column,
        how="left",
        validate="many_to_one",
    )

    numeric_columns = [
        pipe_length_column,
        longitudinal_distance_column,
        normalized_defect_length_column,
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result["Defect_length_m"] = (
        result[normalized_defect_length_column]
        * result[pipe_length_column]
    )

    result["Defect_start_m"] = result[
        longitudinal_distance_column
    ]

    result["Defect_end_m"] = (
        result["Defect_start_m"]
        + result["Defect_length_m"].fillna(0)
    )

    result["Defect_end_m"] = np.minimum(
        result["Defect_end_m"],
        result[pipe_length_column],
    )

    return result

def calculate_peak_score(
    pipe_defects,
    score_column,
    start_column="Defect_start_m",
    end_column="Defect_end_m",
    pipe_length_column="Pipe_length",
    window_length=1.0,
):
    """
    Calculate the maximum defect score accumulated within any
    one-metre section of a pipe.

    Continuous defects contribute their score multiplied by the
    length overlapping the window. Point defects contribute their
    complete score when located inside the window.

    The function assumes that:
    - defect start and end positions have already been calculated;
    - defect start is less than or equal to defect end;
    - continuous defects have start < end;
    - point defects have start == end.

    Parameters
    ----------
    pipe_defects : pandas.DataFrame
        Defects belonging to one pipe.
    score_column : str
        Column containing either structural or service scores.
    window_length : float, default 1.0
        Length of the moving assessment window in metres.

    Returns
    -------
    float
        Maximum accumulated score within any assessment window.
    """
    required_columns = [
        score_column,
        start_column,
        end_column,
        pipe_length_column,
    ]

    data = pipe_defects[required_columns].copy()

    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    pipe_length_values = data[pipe_length_column].dropna()

    if pipe_length_values.empty:
        return np.nan

    pipe_length = float(pipe_length_values.iloc[0])

    if pipe_length <= 0:
        return np.nan

    data = data.dropna(
        subset=[
            score_column,
            start_column,
            end_column,
        ]
    )

    data = data[data[score_column] != 0]

    if data.empty:
        return 0.0

    defect_lengths = (
        data[end_column]
        - data[start_column]
    )

    continuous_defects = data.loc[
        defect_lengths > 0
    ]

    point_defects = data.loc[
        defect_lengths == 0
    ]

    maximum_window_start = max(
        pipe_length - window_length,
        0.0,
    )

    important_positions = set(
        data[start_column].tolist()
    )

    important_positions.update(
        data[end_column].tolist()
    )

    candidate_window_starts = {
        0.0,
        maximum_window_start,
    }

    for position in important_positions:
        candidate_window_starts.add(
            float(
                np.clip(
                    position,
                    0.0,
                    maximum_window_start,
                )
            )
        )

        candidate_window_starts.add(
            float(
                np.clip(
                    position - window_length,
                    0.0,
                    maximum_window_start,
                )
            )
        )

    def calculate_window_score(window_start):
        """Calculate the accumulated score inside one window."""
        window_end = min(
            window_start + window_length,
            pipe_length,
        )

        total_score = 0.0

        if not continuous_defects.empty:
            overlap_start = np.maximum(
                continuous_defects[start_column].to_numpy(
                    dtype=float
                ),
                window_start,
            )

            overlap_end = np.minimum(
                continuous_defects[end_column].to_numpy(
                    dtype=float
                ),
                window_end,
            )

            overlap_lengths = np.maximum(
                overlap_end - overlap_start,
                0.0,
            )

            scores = continuous_defects[
                score_column
            ].to_numpy(dtype=float)

            total_score += np.sum(
                scores * overlap_lengths
            )

        if not point_defects.empty:
            positions = point_defects[
                start_column
            ].to_numpy(dtype=float)

            scores = point_defects[
                score_column
            ].to_numpy(dtype=float)

            inside_window = (
                (positions >= window_start)
                & (positions <= window_end)
            )

            total_score += np.sum(
                scores[inside_window]
            )

        return float(total_score)

    return max(
        calculate_window_score(window_start)
        for window_start in candidate_window_starts
    )

def calculate_pipe_peak_scores(
    defects,
    pipe_id_column="Pipe_ID",
):
    """
    Calculate one structural peak score and one service peak score
    for each pipe.
    """
    records = []

    for pipe_id, pipe_defects in defects.groupby(
        pipe_id_column,
        sort=False,
    ):
        records.append(
            {
                pipe_id_column: pipe_id,
                "Structural_peak_score": calculate_peak_score(
                    pipe_defects,
                    score_column="Structural_score",
                ),
                "Service_peak_score": calculate_peak_score(
                    pipe_defects,
                    score_column="Service_score",
                ),
            }
        )

    return pd.DataFrame(records)

def assign_condition_grade(peak_score):
    """
    Convert a peak score into a preliminary condition grade according
    to the NZGPIM Fourth Edition.
    """
    if pd.isna(peak_score):
        return pd.NA

    if peak_score <= 5:
        return 1

    if peak_score <= 20:
        return 2

    if peak_score <= 35:
        return 3

    if peak_score <= 60:
        return 4

    return 5

def add_condition_grades(peak_scores):
    """
    Add structural and service preliminary condition grades.
    """
    result = peak_scores.copy()

    result["Structural_condition_grade"] = (
        result["Structural_peak_score"]
        .apply(assign_condition_grade)
        .astype("Int64")
    )

    result["Service_condition_grade"] = (
        result["Service_peak_score"]
        .apply(assign_condition_grade)
        .astype("Int64")
    )

    return result

def merge_condition_results(
    cctv,
    peak_scores,
    pipe_id_column="Pipe_ID",
):
    """
    Merge pipe-level peak scores and condition grades into the CCTV
    dataset.

    Pipes without recorded defects are assigned a peak score of zero
    and a preliminary condition grade of one.
    """
    result = cctv.copy()

    output_columns = [
        "Structural_peak_score",
        "Service_peak_score",
        "Structural_condition_grade",
        "Service_condition_grade",
    ]

    existing_columns = [
        column
        for column in output_columns
        if column in result.columns
    ]

    if existing_columns:
        result = result.drop(columns=existing_columns)

    result = result.merge(
        peak_scores,
        on=pipe_id_column,
        how="left",
        validate="many_to_one",
    )

    result[
        [
            "Structural_peak_score",
            "Service_peak_score",
        ]
    ] = result[
        [
            "Structural_peak_score",
            "Service_peak_score",
        ]
    ].fillna(0.0)

    result[
        [
            "Structural_condition_grade",
            "Service_condition_grade",
        ]
    ] = result[
        [
            "Structural_condition_grade",
            "Service_condition_grade",
        ]
    ].fillna(1).astype("Int64")

    return result

def calculate_condition_scores(
    defects,
    cctv,
    pipes,
):
    """
    Calculate structural and service peak scores and preliminary
    condition grades for all CCTV inspections.

    Returns
    -------
    prepared_defects : pandas.DataFrame
        Defect-level dataset with score codes, individual scores,
        and defect positions in metres.

    cctv_with_condition : pandas.DataFrame
        CCTV dataset with one structural and one service peak score
        and condition grade per pipe.
    """

    # ------------------------------------------------------------
    # 1. Validate existing Quantification values
    # ------------------------------------------------------------

    validate_quantification_values(
        defects=defects,
    )

    # ------------------------------------------------------------
    # 2. Fill missing Quantification values where required
    # ------------------------------------------------------------

    defects = size_validation(
        defects,
        structural_scores=_structural_scores,
        service_scores=_service_scores,
    )

    # ------------------------------------------------------------
    # 3. Create score codes and assign structural/service scores
    # ------------------------------------------------------------

    prepared_defects = add_defect_scores(
        defects=defects,
    )

    # ------------------------------------------------------------
    # 4. Check that every score code exists in at least one
    #    score dictionary
    # ------------------------------------------------------------

    unknown_codes = find_unknown_score_codes(
        defects=prepared_defects,
    )

    if not unknown_codes.empty:

        unknown_values = unknown_codes.to_dict(
            orient="records"
        )

        raise ValueError(
            "Undefined defect score codes were found: "
            f"{unknown_values}"
        )

    # ------------------------------------------------------------
    # 5. Add defect positions and pipe lengths
    # ------------------------------------------------------------

    prepared_defects = add_defect_positions(
        defects=prepared_defects,
        pipes=pipes,
    )

    # ------------------------------------------------------------
    # 6. Validate pipe lengths
    # ------------------------------------------------------------

    validate_pipe_lengths(
        defects=prepared_defects,
    )

    # ------------------------------------------------------------
    # 7. Calculate structural and service peak scores
    # ------------------------------------------------------------

    peak_scores = calculate_pipe_peak_scores(
        defects=prepared_defects,
    )

    # ------------------------------------------------------------
    # 8. Convert peak scores to condition grades
    # ------------------------------------------------------------

    peak_scores = add_condition_grades(
        peak_scores=peak_scores,
    )

    # ------------------------------------------------------------
    # 9. Merge results into CCTV dataset
    # ------------------------------------------------------------

    cctv_with_condition = merge_condition_results(
        cctv=cctv,
        peak_scores=peak_scores,
    )

    return prepared_defects, cctv_with_condition