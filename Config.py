import seaborn as sns
from typing import Dict
import matplotlib as mpl
import matplotlib.colors as mcolors
from typing import Sequence, Mapping


# ==========================================================
# FACTOR GROUPS
# ==========================================================

FACTORS_DESIGN_NUM = [
    'Installation_year',
    'Diameter',
    'Pipe_length',
    'Depth',
    'Slope',
    'Properties',
    'Wet_peak_flow_rate',
    'Dry_peak_flow_rate',
    'Road_num',
]

FACTORS_ENVI_NUM = [
    'Mean_annual',
    'GWL_from_pipe',
    'Distance_seawater',
    'Liq_vul_num'
]

FACTORS_DESIGN_CAT = [
    'Land_cover_group',
    'Land_use_group',
    "Sewer_category"
]

FACTORS_ENVI_CAT = [
    'Soil_type'
]

FACTORS_NUM = FACTORS_DESIGN_NUM + FACTORS_ENVI_NUM
FACTORS_CAT = FACTORS_DESIGN_CAT + FACTORS_ENVI_CAT

DEFAULT_FACTORS = FACTORS_NUM + FACTORS_CAT
# ==========================================================
# CATEGORY ORDER AND COLORS
# ==========================================================

def get_categories_colors() -> Dict[str, dict]:

    return {
        "Traffic_group": dict(zip(
            ["No road above", "Low vehicle load", "High vehicle load"],
            sns.color_palette("Oranges", n_colors=4)[1:]
        )),

        "Road": dict(zip(
            ["No", "Yes"],
            sns.color_palette("Reds", n_colors=3)[1:]
        )),

        "Land_use_group": dict(zip(
            ["Business", "Open space", "Residential", "Road area"],
            sns.color_palette("Greens", n_colors=5)[1:]
        )),

        "Land_cover_group": dict(zip(
            ["Vegetation", "Urban", "Mixed"],
            sns.color_palette("Blues", n_colors=4)[1:]
        )),

        "Soil_type": dict(zip(
            ["Boulders", "Sand", "Silt and clay"],
            sns.color_palette("Purples", n_colors=4)[1:]
        )),

        "Liq_vul_num": dict(zip(
            ["Very Low", "Unlikely", "Possible"],
            sns.color_palette(["#FFE680", "#F5D042", "#EDC001"])
        )),
    }


def get_categories_order() -> dict:

    return {
        "Traffic_group": [
            "No road above",
            "Low vehicle load",
            "High vehicle load"
        ],

        "Liq_vul_num": [
            "Very Low",
            "Unlikely",
            "Possible"
        ],
        "Liq_vul": [
            "Very Low",
            "Unlikely",
            "Possible"
        ],

        "Road": [
            "No",
            "Yes"
        ],

        "Land_use_group": [
            "Business",
            "Open space",
            "Residential",
            "Road area",
        ],

        "Land_cover_group": [
            "Vegetation",
            "Urban",
            "Mixed"
        ],

        "Soil_type": [
            "Boulders",
            "Sand",
            "Silt and clay"
        ],

        "Sewer_category": [
            "Transmission",
            "Local"
        ]
    }


# ==========================================================
# DISPLAY NAMES AND UNITS
# ==========================================================

DISPLAY_NAMES = {
    'Age_cctv': 'Age',
    'Installation_year': 'Installation year',
    'Pipe_length': 'Length',
    'Depth': 'Depth',
    'Slope': 'Slope',
    'Properties': 'Service connections',
    'Restaurants': 'Restaurants',
    'Mean_annual': 'Mean annual rainfall',
    'Laundries': 'Laundries',
    'GWL_from_pipe': 'Groundwater level from pipe',
    'Distance_seawater': 'Distance to seawater',
    'Wet_peak_flow_rate': 'Wet max. flow rate',
    'Dry_peak_flow_rate': 'Dry max. flow rate',
    'Diameter': 'Diameter',
    'Traffic_num': 'Traffic volume',
    'Liq_vul_num': 'Liquefaction vulnerability',
    'Liq_vul': 'Liquefaction vulnerability',
    'Land_cover_group': 'Land cover',
    'Land_use_group': 'Land use',
    'Soil_type': 'Soil type',
    'Road_num': 'Road',
    'Defect_code': 'Defect type',
    "Sewer_category": "Sewer category",
}


UNITS = {
    'Installation_year': '-',
    'Pipe_length': 'm',
    'Depth': 'm',
    'Slope': '%',
    'Properties': '-',
    'Mean_annual': 'mm/year',
    'Wet_peak_flow_rate': 'L/s',
    'Dry_peak_flow_rate': 'L/s',
    'Diameter': 'mm',
    'Distance_seawater': 'm',
    'GWL_from_pipe': 'm'
}


# ==========================================================
# MATERIAL GROUPS
# ==========================================================

DEFAULT_MATERIALS = ['AC', 'CONC', 'VC', 'PVC', 'PE']
MATERIALS_RIGID = ['AC', 'CONC', 'VC']
MATERIALS_PLASTIC = ['PVC', 'PE']

# ==========================================================
# MATERIAL COLORS
# ==========================================================

# Base colors defined for the paper
BASE_MATERIAL_COLORS: Mapping[str, str] = {
    'VC': '#299729',
    'AC': '#FF6B4A',
    'PE': '#84B3D6',
    'CONC': '#FFB347',
    'PVC': '#9467bd',
}
DEFAULT_OTHER_COLOR = '#999999'

def build_material_color_map(
        selected_materials: Sequence[str],
        base_colors: Mapping[str, str] = BASE_MATERIAL_COLORS,
        other_color: str = DEFAULT_OTHER_COLOR
) -> dict:
    """
    Create a color mapping for selected pipe materials.

    Materials listed in `base_colors` get predefined colors.
    New/unlisted materials receive colors from a colormap.
    An 'OTHERS' entry is always included.

    Parameters
    ----------
    selected_materials : list-like of str
        Materials to assign colors to.
    base_colors : dict, optional
        Predefined colors for known materials.
    other_color : str, optional
        Hex color for the 'OTHERS' category.

    Returns
    -------
    dict
        Mapping material -> color.
    """

    selected_materials = list(selected_materials)

    # Materials that already have predefined colors
    predefined = {
        mat: col for mat, col in base_colors.items()
        if mat in selected_materials
    }

    # Materials that still need colors
    remaining = [m for m in selected_materials if m not in predefined]

    # Generate a colormap for the remaining materials
    cmap = mpl.colormaps['Paired'].resampled(max(1, len(remaining)))

    dynamic_colors = {
        m: mcolors.to_hex(cmap(i))
        for i, m in enumerate(remaining)
    }

    # Combine dictionaries
    colors = {**predefined, **dynamic_colors}

    # Always include "OTHERS"
    colors['OTHERS'] = other_color

    return colors
# ==========================================================
# ANALYSIS DEFAULTS
# ==========================================================

DEFAULT_MIN_OBSERVATIONS_PER_GROUP = 20
DEFAULT_SIGNIFICANCE_THRESHOLD = 0.05

# ==========================================================
# STANDARDIZE CATEGORY NAMES
# ==========================================================

CATEGORY_LABELS = {
    "Road_num": {
        0: "No",
        1: "Yes",
        0.0: "No",
        1.0: "Yes"
    }
}

# ==========================================================
# CONDITION SCORES
# ==========================================================
# Structural defect scores
_structural_scores = {
    "CCL": 12,
    "CCM": 7,
    "CCS": 2,
    "CLL": 21,
    "CLM": 15,
    "CLS": 3,
    "CML": 30,
    "CMM": 25,
    "CMS": 10,
    "DFL": 100,
    "DFM": 5,
    "DPL": 26,
    "DPM": 6,
    "DPS": 1,
    "IPL": 10,
    "IPM": 10,
    "IPS": 1,
    "JDL": 20,
    "JDM": 6,
    "JDS": 1,
    "JFL": 17,
    "JFM": 10,
    "JFS": 4,
    "JOL": 15,
    "JOM": 5,
    "JOS": 1,
    "LFL": 15,
    "LFM": 6,
    "LFS": 2,
    "LXL": 30,
    "LXM": 20,
    "LXS": 5,
    "PBL": 100,
    "PBM": 51,
    "PBS": 30,
    "PFL": 90,
    "PFM": 50,
    "PFS": 25,
    "PHL": 90,
    "PHM": 40,
    "PHS": 25,
    "PLL": 50,
    "PLM": 25,
    "PLS": 5,
    "RIL": 10,
    "RIM": 10,
    "RIS": 3,
    "SL": 61,
    "SM": 21,
    "SS": 6,
    "TM": 40,
    "MHJ": 10,
    "SV": 30,
    "PX": 165,
    "EX": 10,
}

# Service defect scores
_service_scores = {
    "B": 100,
    "DEL": 60,
    "DEM": 35,
    "DES": 15,
    "DFL": 25,
    "DFM": 6,
    "DGL": 55,
    "DGM": 30,
    "DGS": 10,
    "DPL": 60,
    "DPM": 35,
    "DPS": 6,
    "EDL": 55,
    "EDM": 30,
    "EDS": 10,
    "IPL": 30,
    "IPM": 21,
    "IPS": 6,
    "JDL": 36,
    "JDM": 22,
    "JDS": 8,
    "JFL": 6,
    "JFM": 1,
    "JFS": 1,
    "JOL": 8,
    "JOM": 6,
    "JOS": 2,
    "OL": 60,
    "OM": 35,
    "OS": 15,
    "PBL": 6,
    "PBM": 2,
    "PBS": 2,
    "PFL": 25,
    "PFM": 10,
    "PFS": 1,
    "PHL": 20,
    "PHM": 10,
    "PHS": 3,
    "PLL": 40,
    "PLM": 27,
    "PLS": 8,
    "RIL": 55,
    "RIM": 30,
    "RIS": 10,
    "SL": 23,
    "SM": 8,
    "SS": 6,
    "MHJ": 8,
    "PX": 165,
    "EX": 10,
    "LPL": 15,
    "LPM": 10,
    "LPS": 3,
}

VALID_QUANTIFICATIONS = {"L", "M", "S"}