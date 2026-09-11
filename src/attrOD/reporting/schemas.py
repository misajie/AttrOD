"""Column contracts for Tables 1–8 (draft2). Study-area rows left empty — do not invent cities."""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

TABLE_SCHEMAS: Dict[str, List[str]] = {
    "table1_study_areas": [
        "study_area", "product", "N", "h_intra", "days", "attributes_present",
    ],
    "table2_purpose_time": [
        "study_area", "partition", "lambda",
        "CPC_T_lambdaR", "CPC_T_lambdaRT", "Delta",
        "CPL", "CPCd", "R2", "NRMSE", "JSD",
    ],
    "table3_day_sex_age": [
        "study_area", "partition", "lambda",
        "CPC_T_lambdaR", "Delta", "CPL", "CPCd", "R2", "NRMSE", "JSD",
    ],
    "table4_length_bands": [
        "study_area", "band", "CPC", "CPL", "CPCd", "R2", "NRMSE", "JSD",
    ],
    "table5_additional": [
        "study_area", "partition", "CPC", "Delta", "CPL", "CPCd", "R2", "NRMSE", "JSD",
    ],
    "table6_condition_g_blocks": [
        "study_area", "target", "law_model",
        "CPC", "CPL", "CPCd", "R2", "NRMSE", "JSD",
    ],
    "table7_panel_no_holdout": [
        "study_area", "target", "law_model",
        "CPC", "CPL", "CPCd", "R2", "NRMSE", "JSD",
    ],
    "table8_scale": [
        "study_area", "scale_layer", "spearman_CPC", "frac_delta_sign_kept",
    ],
}

# Primary purpose–time partition labels for Table 2
TABLE2_PARTITIONS = ["other|AM", "home|PM", "home|AM"]

# Table 6 model row order
TABLE6_MODELS = [
    "Gravity_power",
    "Gravity_exp",
    "Radiation",
    "DeepGravity",
    "ClosedFormGravity",
    "meta-Gravity",
    "neuroGravity_zeroshot",
    "neuroGravity_1pct_edges",
    "neuroGravity_10pct_edges",
    "RandomForest",
    "IPF_G-row",
    "Oracle",
]


def empty_table(name: str) -> pd.DataFrame:
    if name not in TABLE_SCHEMAS:
        raise KeyError(name)
    return pd.DataFrame(columns=TABLE_SCHEMAS[name])
