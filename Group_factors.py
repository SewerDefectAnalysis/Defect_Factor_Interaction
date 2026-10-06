
import pandas as pd
import numpy as np

# ============================================================
# 1. EQUAL-WIDTH GROUPING FOR NUMERICAL VARIABLES
# ============================================================

def create_equal_range_groups(
    df,
    column_name,
    range_fraction=0.05,
    decimals=None
):
    """
    Divide a numerical variable into equal-width intervals based on
    its observed minimum and maximum values.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe.

    column_name : str
        Numerical column to group.

    range_fraction : float, default 0.05
        Fraction of the total observed range represented by each group.
        For example:
            0.05 -> 20 groups
            0.10 -> 10 groups
            0.20 -> 5 groups

    decimals : int or None
        Number of decimal places used in the group labels.
        If None, it is selected automatically.

    Returns
    -------
    pd.Series
        Categorical series containing labels based on the actual
        numerical limits of each interval.
    """

    if not 0 < range_fraction <= 1:
        raise ValueError(
            "range_fraction must be greater than 0 and less than or equal to 1."
        )

    # Convert the selected variable to numeric
    values = pd.to_numeric(
        df[column_name],
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    )

    # No valid observations
    if values.notna().sum() == 0:
        return pd.Series(
            pd.NA,
            index=df.index,
            dtype="category"
        )

    min_value = values.min()
    max_value = values.max()
    total_range = max_value - min_value

    # All observations have the same value
    if pd.isna(total_range) or total_range == 0:
        output = pd.Series(
            pd.NA,
            index=df.index,
            dtype="object"
        )

        output.loc[values.notna()] = f"{min_value:g}"

        return output.astype("category")

    # Number of equal-width groups
    n_groups = int(np.ceil(1 / range_fraction))

    # Create equally spaced limits
    bins = np.linspace(
        min_value,
        max_value,
        n_groups + 1
    )

    bins = np.unique(bins)

    if len(bins) < 2:
        return pd.Series(
            pd.NA,
            index=df.index,
            dtype="category"
        )

    # Automatically determine label precision
    if decimals is None:
        step = bins[1] - bins[0]

        if step >= 1:
            decimals = 0
        elif step >= 0.1:
            decimals = 1
        elif step >= 0.01:
            decimals = 2
        else:
            decimals = 3

    def format_value(value):
        if decimals == 0:
            return f"{value:.0f}"

        return f"{value:.{decimals}f}"

    # Labels using actual values
    labels = [
        f"{format_value(bins[i])}-{format_value(bins[i + 1])}"
        for i in range(len(bins) - 1)
    ]

    output = pd.Series(
        pd.NA,
        index=df.index,
        dtype="object"
    )

    valid_mask = values.notna()

    grouped = pd.cut(
        values.loc[valid_mask],
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
        duplicates="drop"
    )

    output.loc[valid_mask] = grouped.astype("object")

    return output.astype(
        pd.CategoricalDtype(
            categories=labels,
            ordered=True
        )
    )

# ============================================================
# 2. SPECIAL GROUPING FOR DIAMETER
# ============================================================

def create_diameter_groups(
    df,
    column_name="Diameter"
):
    """
    Assign each observed diameter to the nearest representative
    commercial diameter and then group adjacent commercial diameters.
    """

    values = pd.to_numeric(
        df[column_name],
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Representative commercial diameters
    commercial_diameters = np.array([
        100,
        150,
        175,
        200,
        225,
        250,
        300,
        375,
        450,
        525,
        600,
        675,
        750,
        800,
        900,
        1050,
        1200,
        1350,
        1500,
        1650,
        1800,
        1950,
        2100,
        2300,
        2550
    ])

    # --------------------------------------------------------
    # Find nearest commercial diameter
    # --------------------------------------------------------

    def nearest_commercial_diameter(value):

        if pd.isna(value):
            return np.nan

        idx = np.abs(
            commercial_diameters - value
        ).argmin()

        return commercial_diameters[idx]

    commercial = values.apply(
        nearest_commercial_diameter
    )

    # --------------------------------------------------------
    # Group adjacent commercial diameters
    # --------------------------------------------------------

    diameter_group_map = {

        100: "100-150",
        150: "100-150",

        175: "175-225",
        200: "175-225",
        225: "175-225",

        250: "250-300",
        300: "250-300",

        375: "375-450",
        450: "375-450",

        525: "525-600",
        600: "525-600",

        675: "675-750",
        750: "675-750",

        800: "800-900",
        900: "800-900",

        1050: "1050-1200",
        1200: "1050-1200",

        1350: "1350-1500",
        1500: "1350-1500",

        1650: "1650-1800",
        1800: "1650-1800",

        1950: "1950-2100",
        2100: "1950-2100",

        2300: "2300-2550",
        2550: "2300-2550"
    }

    labels = [
        "100-150",
        "175-225",
        "250-300",
        "375-450",
        "525-600",
        "675-750",
        "800-900",
        "1050-1200",
        "1350-1500",
        "1650-1800",
        "1950-2100",
        "2300-2550"
    ]

    grouped = commercial.map(
        diameter_group_map
    )

    return grouped.astype(
        pd.CategoricalDtype(
            categories=labels,
            ordered=True
        )
    )


# ============================================================
# 3. CREATE ALL GROUPED VARIABLES
# ============================================================
categorical_variables = [
    "Road",
    "Liq_vul_num"
]

def create_grouped_variables(
    df,
    variables,
    range_fraction=0.05,
    decimals_mapping=None,
    categorical_variables=categorical_variables,
    suffix="_group"
):
    """
    Create grouped versions of variables.

    Grouping rules:
    - Diameter: grouped according to representative commercial diameters.
    - Categorical/binary variables: values are retained as their original categories.
    - Other numerical variables: grouped using equal-width intervals.
    """

    grouped_df = df.copy()

    decimals_mapping = decimals_mapping or {}

    if categorical_variables is None:
        categorical_variables = []

    for variable in variables:

        if variable not in grouped_df.columns:

            print(
                f"Warning: '{variable}' was not found in the dataframe."
            )

            continue

        grouped_column = f"{variable}{suffix}"

        # ----------------------------------------------------
        # Diameter: commercial diameter groups
        # ----------------------------------------------------
        if variable == "Diameter":

            grouped_df[grouped_column] = create_diameter_groups(
                df=grouped_df,
                column_name=variable
            )

        # ----------------------------------------------------
        # Variables that are already categorical
        # ----------------------------------------------------
        elif variable in categorical_variables:

            grouped_df[grouped_column] = (
                grouped_df[variable]
                .astype("category")
            )

        # ----------------------------------------------------
        # Continuous numerical variables
        # ----------------------------------------------------
        else:

            grouped_df[grouped_column] = create_equal_range_groups(
                df=grouped_df,
                column_name=variable,
                range_fraction=range_fraction,
                decimals=decimals_mapping.get(variable)
            )

    return grouped_df


group_cols = {
    'Installation_year': 'Installation_year_group',
    'Pipe_length': 'Pipe_length_group',
    'Diameter': 'Diameter_group',
    'Depth':'Depth_group',
    'Slope':'Slope_group',
    'Wet_peak_flow_rate':'Wet_peak_flow_rate_group',
    'Properties':'Properties_group',
    'Restaurants':'Restaurants_group',
    'GWL_from_pipe':'GWL_from_pipe_group',
    'Mean_annual':'Mean_annual_group',
    'Distance_seawater':'Distance_seawater_group',
    'Road':'Road_group',
    'Liq_vul_num':'Liq_vul_num_group',
    'Dry_peak_flow_rate':'Dry_peak_flow_rate_group',
}