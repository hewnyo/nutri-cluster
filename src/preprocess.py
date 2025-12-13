# src/preprocess.py
import os
import re
import pandas as pd

TEXT_COLS = ["SKLL_IX_IRDNT_RAWMTRL", "PRIMARY_FNCLTY"]

META_COLS = [
    "PRDCT_NM",
    "PRIMARY_FNCLTY",
    "SKLL_IX_IRDNT_RAWMTRL",
    "IFTKN_ATNT_MATR_CN",
    "INTK_UNIT",
    "DAY_INTK_LOWLIMIT",
    "DAY_INTK_HIGHLIMIT",
]

# ✅ I2710(기능성 원료/기능성 설명) 특성에 맞춘 "장 건강/배변" 키워드 중심 사전
KEYWORDS = {
    # 장/배변/변비
    "gut_health": r"(장\s*건강|장내|장\s*기능|intestinal|gut)",
    "bowel_movement": r"(배변|변비|장\s*활동|화장실|배변활동|배변\s*활동)",
    "digestion": r"(소화|digest|소화기|위장)",

    # 식이섬유 계열
    "dietary_fiber": r"(식이섬유|fiber|화이버)",
    "acacia_arabic_gum": r"(아라비아검|아카시아검|arabic\s*gum|acacia)",
    "psyllium": r"(차전자피|psyllium)",
    "inulin": r"(이눌린|inulin)",
    "fodmap_oligo": r"(프락토올리고당|올리고당|oligosaccharide)",

    # 유산균/프리바이오틱스(있으면 잡히게)
    "probiotics": r"(프로바이오틱스|유산균|락토바실러스|비피더스|probiotic|lactobac|bifido)",
    "prebiotics": r"(프리바이오틱스|prebiotic)",

    # (선택) 혈당/콜레스테롤 같은 대사 관련도 장 건강과 함께 나올 수 있음
    "blood_glucose": r"(혈당|glucose)",
    "cholesterol": r"(콜레스테롤|cholesterol)",
}

def _normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _merged_text(df: pd.DataFrame) -> pd.Series:
    merged = pd.Series([""] * len(df), index=df.index, dtype="object")
    for c in TEXT_COLS:
        if c in df.columns:
            merged = merged + " " + df[c].fillna("").astype(str)
    return merged.apply(_normalize_text)

def build_feature_vector(df: pd.DataFrame, keywords: dict = None) -> pd.DataFrame:
    if keywords is None:
        keywords = KEYWORDS

    text = _merged_text(df)
    features = pd.DataFrame(index=df.index)

    for feat, pattern in keywords.items():
        features[feat] = text.str.contains(pattern, regex=True).astype(int)

    # 섭취량(가능하면 숫자로 포함) — 범위 기반 클러스터링에도 도움됨
    for col in ["DAY_INTK_LOWLIMIT", "DAY_INTK_HIGHLIMIT"]:
        if col in df.columns:
            features[col] = pd.to_numeric(df[col], errors="coerce")

    return features

def preprocess_for_gut_reco(df: pd.DataFrame):
    """
    returns:
      features_df: (0/1 키워드 + 섭취량) 벡터
      meta_df: 해석용
    """
    df = df.copy()

    features = build_feature_vector(df)

    # ✅ 키워드가 하나도 안 잡힌 행은 제거 (원하면 주석처리 가능)
    keyword_cols = [c for c in features.columns if c not in ["DAY_INTK_LOWLIMIT", "DAY_INTK_HIGHLIMIT"]]
    mask = features[keyword_cols].sum(axis=1) > 0

    features_df = features.loc[mask].reset_index(drop=True)
    meta_df = df.loc[mask, [c for c in META_COLS if c in df.columns]].reset_index(drop=True)

    return features_df, meta_df

def save_processed(features_df: pd.DataFrame, meta_df: pd.DataFrame, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    features_path = os.path.join(out_dir, "features.csv")
    meta_path = os.path.join(out_dir, "meta.csv")

    features_df.to_csv(features_path, index=False, encoding="utf-8-sig")
    meta_df.to_csv(meta_path, index=False, encoding="utf-8-sig")

    print("✅ Saved:")
    print(" -", features_path)
    print(" -", meta_path)
