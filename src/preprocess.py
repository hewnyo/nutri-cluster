# src/preprocess.py
import os
import re
import pandas as pd

# C003에서 텍스트로 볼 만한 컬럼 후보들(실제 존재하는 것만 합쳐서 사용)
TEXT_CANDIDATES = [
    "PRDLST_NM", "PRDT_NM",
    "RAWMTRL_NM", "RAWMTRL", "RAWMTRL_CN",
    "PRIMARY_FNCLTY", "SKLL_IX_IRDNT_RAWMTRL",
    "IFTKN_ATNT_MATR_CN",
]

META_KEEP = [
    "PRDLST_NM", "BSSH_NM", "PRMS_DT", "CHNG_DT",
    "PRDLST_REPORT_NO", "LCNS_NO"
]

# ✅ 성분(영양/기능성 성분) 키워드 사전: 포함 여부(0/1) 벡터로 만들 것
NUTRI_KEYWORDS = {
    "vitamin_c": r"(비타민\s*c|vitamin\s*c|ascorbic)",
    "vitamin_b": r"(비타민\s*b|vitamin\s*b|b1|b2|b6|b12|비오틴|나이아신|판토텐산|엽산)",
    "vitamin_d": r"(비타민\s*d|vitamin\s*d)",
    "zinc": r"(아연|zinc)",
    "magnesium": r"(마그네슘|magnesium)",
    "iron": r"(철|iron)",
    "folate": r"(엽산|folate)",
    "selenium": r"(셀레늄|selenium)",

    "probiotics": r"(프로바이오틱|유산균|lactobac|bifido|probiotic)",
    "prebiotics": r"(프리바이오틱|이눌린|inulin|프락토올리고당|올리고당)",

    "lutein": r"(루테인|lutein|지아잔틴|zeaxanthin)",
    "astaxanthin": r"(아스타잔틴|astaxanthin)",

    "collagen": r"(콜라겐|collagen)",
    "hyaluronic": r"(히알루론산|hyaluronic)",

    "calcium": r"(칼슘|calcium)",

    "msm": r"(msm|엠에스엠)",
    "glucosamine": r"(글루코사민|glucosamine)",
    "chondroitin": r"(콘드로이친|chondroitin)",

    "omega3": r"(오메가\s*3|omega\s*3|epa|dha)",
    "coq10": r"(코엔자임\s*q10|coq10|유비퀴논)",
    "milk_thistle": r"(밀크씨슬|실리마린|silymarin)",
    "red_ginseng": r"(홍삼|red\s*ginseng|진세노사이드)",
    "l_theanine": r"(l-?테아닌|theanine)",
    "melatonin": r"(멜라토닌|melatonin)",
    "garcinia": r"(가르시니아|garcinia)",
    "green_tea": r"(녹차|green\s*tea|카테킨|catechin)",
}

# ✅ 니즈(사용자 상태) → 어떤 성분을 선호할지(가중치) 매핑
NEED_TO_NUTRI = {
    "fatigue": {  # 피로/기력
        "vitamin_c": 2,
        "vitamin_b": 3,
        "magnesium": 1,
        "iron": 1,
        "coq10": 2,
        "red_ginseng": 3,
    },
    "immune": {  # 면역/항산화
        "vitamin_c": 3,
        "vitamin_d": 2,
        "zinc": 2,
        "selenium": 1,
        "red_ginseng": 2,
    },
    "sleep": {   # 수면/긴장
        "melatonin": 3,
        "l_theanine": 2,
        "magnesium": 1,
    },
    "gut": {     # 장 건강
        "probiotics": 3,
        "prebiotics": 2,
    },
    "eye": {     # 눈
        "lutein": 3,
        "astaxanthin": 2,
    },
    "liver": {   # 간
        "milk_thistle": 3,
    },
    "joint": {   # 관절
        "glucosamine": 2,
        "chondroitin": 2,
        "msm": 1,
    },
    "diet": {    # 체지방/다이어트
        "garcinia": 2,
        "green_tea": 2,
    },
    "skin": {    # 피부
        "collagen": 2,
        "hyaluronic": 2,
        "vitamin_c": 1,
    }
}

def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s

def _merge_text(df: pd.DataFrame) -> pd.Series:
    merged = pd.Series([""] * len(df), index=df.index, dtype="object")
    for c in TEXT_CANDIDATES:
        if c in df.columns:
            merged = merged + " " + df[c].fillna("").astype(str)
    return merged.apply(_norm)

def preprocess_for_reco(df_raw: pd.DataFrame):
    """
    returns:
      features_df: 성분 포함 여부(0/1) + (선택) 텍스트 길이 등
      meta_df: 해석용 메타
    """
    df = df_raw.copy()

    # 날짜 파싱(있으면)
    for c in ["PRMS_DT", "CHNG_DT"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    text = _merge_text(df)

    features = pd.DataFrame(index=df.index)
    for feat, pattern in NUTRI_KEYWORDS.items():
        features[feat] = text.str.contains(pattern, regex=True).astype(int)

    # ❗전부 0인 행 제거(원하면 주석처리 가능)
    mask = features.sum(axis=1) > 0
    features_df = features.loc[mask].reset_index(drop=True)

    keep = [c for c in META_KEEP if c in df.columns]
    if not keep:
        keep = [c for c in ["PRDLST_NM", "BSSH_NM"] if c in df.columns]
    meta_df = df.loc[mask, keep].reset_index(drop=True)

    return features_df, meta_df

def recommend_by_need(features_df: pd.DataFrame, meta_df: pd.DataFrame, need: str, top_n: int = 10):
    """
    need: 'fatigue', 'immune', 'sleep', 'gut', ...
    """
    if need not in NEED_TO_NUTRI:
        raise ValueError(f"need must be one of {list(NEED_TO_NUTRI.keys())}")

    weights = NEED_TO_NUTRI[need]

    # score = Σ (feature * weight)
    score = pd.Series(0, index=features_df.index, dtype="int64")
    for k, w in weights.items():
        if k in features_df.columns:
            score += features_df[k].fillna(0).astype(int) * int(w)

    out = meta_df.copy()
    out["score"] = score.values
    out = out.sort_values("score", ascending=False).head(top_n)
    return out

def save_processed(features_df: pd.DataFrame, meta_df: pd.DataFrame, out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    features_df.to_csv(os.path.join(out_dir, "features_reco.csv"), index=False, encoding="utf-8-sig")
    meta_df.to_csv(os.path.join(out_dir, "meta_reco.csv"), index=False, encoding="utf-8-sig")
    print("✅ Saved to", out_dir)
