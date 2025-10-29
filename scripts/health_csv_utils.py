"""Utilities for converting cardiovascular risk CSV data into instruction datasets."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

__all__ = [
    "CARDIO_INSTRUCTION_TEXT",
    "DEFAULT_FEATURE_COLUMNS",
    "CATEGORICAL_FEATURE_COLUMNS",
    "safe_value",
    "describe_health_status",
    "build_tabular_features",
    "get_categorical_feature_indices",
    "convert_row_to_example",
    "load_health_dataset_from_csv",
]

CARDIO_INSTRUCTION_TEXT = (
    "你是一名心血管疾病风险预测助手。"
    "根据给定肿瘤患者的描述，判断该患者未来五年是否会发生冠心病。"
    "请严格遵循以下要求：\n"
    "1. 仅输出“是”或“否”；\n"
    "2. 不得生成解释、推理过程或其他文字；\n"
    "3. 输出应仅包含预测结果。\n"
)

DEFAULT_FEATURE_COLUMNS: List[str] = [
    "21022-0.0",  # age
    "31-0.0",  # sex
    "26227-0.0",  # prs_cad
    "26223-0.0",  # prs_cvd
    "23111-0.0",  # leg fat percentage
    "87-0.0",  # illness age
    "135-0.0",  # illness count
    "family_hd",  # family history
    "20161-0.0",  # pack years
    "924-0.0",  # walking speed categorical
    "137-0.0",  # meds count
    "HYPT",  # hypertension history
    "23125-0.0",  # arm mass
    "23110-0.0",  # arm impedance
    "30750-0.0",  # HbA1c
    "30880-0.0",  # urate
]

"""Columns that should be treated as categorical when training TabPFN."""
CATEGORICAL_FEATURE_COLUMNS: List[str] = [
    "31-0.0",  # sex
    "family_hd",  # family history
    "924-0.0",  # walking speed
    "HYPT",  # hypertension history
]

_MAPPING_FAMILY_HD = {0: "无家族性心脏病史", 1: "有家族性心脏病史"}
_MAPPING_HYPT = {0: "无高血压病史", 1: "有高血压病史"}
_MAPPING_WALK = {0: "步行速度慢", 1: "步行速度中等", 2: "步行速度快"}
_MAPPING_LABEL = {0: "否", 1: "是"}


def safe_value(value: Any, digits: int = 2) -> Optional[Any]:
    """Return a rounded numeric value or ``None`` when the input is missing."""

    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, (float, int)):
        float_value = float(value)
        if math.isnan(float_value):
            return None
        return round(float_value, digits)
    if pd.isna(value):  # handles pandas-specific NA types
        return None
    return value


def _safe_numeric(value: Any) -> float:
    """Return a float value for tabular features, preserving missing entries as ``nan``."""

    cleaned = safe_value(value, digits=6)
    if cleaned is None:
        return float("nan")
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return float("nan")


def describe_health_status(row: Mapping[str, Any]) -> str:
    """Generate qualitative health commentary based on selected biomarkers."""

    desc: List[str] = []

    hba1c = safe_value(row.get("30750-0.0"))
    if hba1c is not None:
        if hba1c >= 6.5:
            desc.append("糖化血红蛋白偏高，提示可能存在糖尿病或血糖控制不佳。")
        elif hba1c >= 5.7:
            desc.append("糖化血红蛋白略高，血糖控制需关注。")
        else:
            desc.append("糖化血红蛋白正常。")

    urate = safe_value(row.get("30880-0.0"))
    if urate is not None:
        if urate > 420:
            desc.append("尿酸水平升高。")
        elif urate < 180:
            desc.append("尿酸水平偏低。")
        else:
            desc.append("尿酸水平正常。")

    walk = row.get("924-0.0")
    if walk == 0:
        desc.append("步行速度较慢，体能状况可能较差。")
    elif walk == 2:
        desc.append("步行速度较快，体能状况良好。")

    return " ".join(desc)


def build_tabular_features(
    row: Mapping[str, Any],
    *,
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
) -> List[float]:
    """Extract ordered numeric features for TabPFN from a CSV row."""

    return [_safe_numeric(row.get(column)) for column in feature_columns]


def get_categorical_feature_indices(feature_columns: Sequence[str]) -> List[int]:
    """Return indices of categorical feature columns within ``feature_columns``."""

    categorical_set = set(CATEGORICAL_FEATURE_COLUMNS)
    return [idx for idx, column in enumerate(feature_columns) if column in categorical_set]


def convert_row_to_example(
    row: Mapping[str, Any],
    *,
    tab_features_field: str = "tab_features",
    feature_columns: Sequence[str] = DEFAULT_FEATURE_COLUMNS,
    require_label: bool = False,
) -> Optional[Dict[str, Any]]:
    """Convert a single CSV row to an instruction-example dictionary."""

    age = safe_value(row.get("21022-0.0"))
    sex_raw = row.get("31-0.0")
    if sex_raw == 1:
        sex = "男性"
    elif sex_raw == 0:
        sex = "女性"
    else:
        sex = "未知"

    prs_cad = safe_value(row.get("26227-0.0"))
    prs_cvd = safe_value(row.get("26223-0.0"))
    leg_fat = safe_value(row.get("23111-0.0"))
    illness_age = safe_value(row.get("87-0.0"))
    illness_count = safe_value(row.get("135-0.0"))
    family_hd = _MAPPING_FAMILY_HD.get(row.get("family_hd"), "家族史未知")
    pack_years = safe_value(row.get("20161-0.0"))
    walk = _MAPPING_WALK.get(row.get("924-0.0"), "步行速度未知")
    meds = safe_value(row.get("137-0.0"))
    hypt = _MAPPING_HYPT.get(row.get("HYPT"), "高血压病史未知")
    arm_mass = safe_value(row.get("23125-0.0"))
    arm_imp = safe_value(row.get("23110-0.0"))
    hba1c = safe_value(row.get("30750-0.0"))
    urate = safe_value(row.get("30880-0.0"))

    health_status = describe_health_status(row)

    input_text = (
        f"该患者为{age if age is not None else '未知年龄'}岁{sex}肿瘤患者。"
        f"冠心病多基因风险评分为{prs_cad if prs_cad is not None else '未知'}，"
        f"心血管疾病多基因风险评分为{prs_cvd if prs_cvd is not None else '未知'}。"
        f"右腿脂肪百分比为{leg_fat if leg_fat is not None else '未知'}%。"
        f"非癌症疾病首次发生年龄为{illness_age if illness_age is not None else '未知'}岁，"
        f"共有{illness_count if illness_count is not None else '未知'}种非癌症疾病。"
        f"{family_hd}。吸烟包年数为{pack_years if pack_years is not None else '未知'}。"
        f"平时步行速度为{walk}。目前服用{meds if meds is not None else '未知'}种药物，{hypt}。"
        f"左臂去脂体质量为{arm_mass if arm_mass is not None else '未知'}kg，"
        f"左臂阻抗为{arm_imp if arm_imp is not None else '未知'}Ω。"
        f"糖化血红蛋白(HbA1c)为{hba1c if hba1c is not None else '未知'}%，"
        f"尿酸水平为{urate if urate is not None else '未知'}μmol/L。"
        f"{health_status}"
    )

    label_value = row.get("Label_v2")
    label_text = _MAPPING_LABEL.get(label_value)
    if label_text is None:
        if require_label:
            return None
    example: Dict[str, Any] = {
        "instruction": CARDIO_INSTRUCTION_TEXT,
        "input": input_text,
    }
    if label_text is not None:
        example["output"] = label_text
        example["label"] = label_text
    example[tab_features_field] = build_tabular_features(row, feature_columns=feature_columns)
    return example


def load_health_dataset_from_csv(
    csv_path: str | Path,
    *,
    tab_features_field: str = "tab_features",
    feature_columns: Optional[Sequence[str]] = None,
    require_label: bool = False,
) -> Tuple[List[Dict[str, Any]], Optional[str], List[str], List[str]]:
    """Load a CSV file and convert it into instruction examples."""

    df = pd.read_csv(csv_path)
    columns = list(feature_columns) if feature_columns is not None else DEFAULT_FEATURE_COLUMNS
    categorical_columns = [
        column for column in columns if column in CATEGORICAL_FEATURE_COLUMNS
    ]

    examples: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        example = convert_row_to_example(
            row,
            tab_features_field=tab_features_field,
            feature_columns=columns,
            require_label=require_label,
        )
        if example is not None:
            examples.append(example)

    if not examples:
        raise ValueError(f"No usable rows found in CSV file: {csv_path}")

    label_field = "label" if any("label" in ex for ex in examples) else None
    return examples, label_field, columns, categorical_columns
