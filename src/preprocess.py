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
# (캡처 그룹 경고 방지를 위해 ( ... ) 대신 (?: ... ) 사용)
NUTRI_KEYWORDS = {
    "vitamin_c": r"(?:비타민\s*c|vitamin\s*c|ascorbic)",
    "vitamin_b": r"(?:비타민\s*b|vitamin\s*b|b1|b2|b6|b12|비오틴|나이아신|판토텐산|엽산)",
    "vitamin_d": r"(?:비타민\s*d|vitamin\s*d)",
    "zinc": r"(?:아연|zinc)",
    "magnesium": r"(?:마그네슘|magnesium)",
    "iron": r"(?:철|iron)",
    "folate": r"(?:엽산|folate)",
    "selenium": r"(?:셀레늄|selenium)",

    "probiotics": r"(?:프로바이오틱|유산균|lactobac|bifido|probiotic)",
    "prebiotics": r"(?:프리바이오틱|이눌린|inulin|프락토올리고당|올리고당)",

    "lutein": r"(?:루테인|lutein|지아잔틴|zeaxanthin)",
    "astaxanthin": r"(?:아스타잔틴|astaxanthin)",

    "collagen": r"(?:콜라겐|collagen)",
    "hyaluronic": r"(?:히알루론산|hyaluronic)",

    "calcium": r"(?:칼슘|calcium)",

    "msm": r"(?:msm|엠에스엠)",
    "glucosamine": r"(?:글루코사민|glucosamine)",
    "chondroitin": r"(?:콘드로이친|chondroitin)",

    "omega3": r"(?:오메가\s*3|omega\s*3|epa|dha)",
    "coq10": r"(?:코엔자임\s*q10|coq10|유비퀴논)",
    "milk_thistle": r"(?:밀크씨슬|실리마린|silymarin)",
    "red_ginseng": r"(?:홍삼|red\s*ginseng|진세노사이드)",
    "l_theanine": r"(?:l-?테아닌|theanine)",
    "melatonin": r"(?:멜라토닌|melatonin)",
    "garcinia": r"(?:가르시니아|garcinia)",
    "green_tea": r"(?:녹차|green\s*tea|카테킨|catechin)",
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

def preprocess_for_reco(df_raw: pd.DataFrame, return_meta: bool = True):
    """
    전처리 (추천용)
    - 텍스트 컬럼들을 합쳐서 키워드 포함 여부(0/1) feature 생성
    - 전부 0인 행 제거

    Args:
      df_raw: API 원본 DataFrame
      return_meta: True면 (features_df, meta_df) 반환 / False면 features_df만 반환

    Returns:
      return_meta=True  -> (features_df, meta_df)
      return_meta=False -> features_df
    """
    df = df_raw.copy()

    # 날짜 파싱(있으면)
    for c in ["PRMS_DT", "CHNG_DT"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    text = _merge_text(df)

    features = pd.DataFrame(index=df.index)
    for feat, pattern in NUTRI_KEYWORDS.items():
        # regex=True 기본, (?:...)로 경고 없이 동작
        features[feat] = text.str.contains(pattern, regex=True).astype(int)

    # ❗전부 0인 행 제거(원하면 주석처리 가능)
    mask = features.sum(axis=1) > 0
    features_df = features.loc[mask].reset_index(drop=True)

    if not return_meta:
        return features_df

    keep = [c for c in META_KEEP if c in df.columns]
    if not keep:
        keep = [c for c in ["PRDLST_NM", "BSSH_NM"] if c in df.columns]
    meta_df = df.loc[mask, keep].reset_index(drop=True)

    return features_df, meta_df

def validate_preprocessed(features_df: pd.DataFrame, meta_df: pd.DataFrame | None = None, top_na_cols: int = 20):
    """
    전처리 결과 품질 리포트 출력 (노트북에서 바로 확인용)
    """
    print("\n" + "=" * 70)
    print("[PREPROCESS VALIDATION REPORT]")
    print("=" * 70)

    # 기본 정보
    print("features_df shape:", features_df.shape)
    print("features_df columns:", list(features_df.columns))
    print("features_df head:")
    print(features_df.head(3))

    # 결측치
    na = features_df.isna().sum().sort_values(ascending=False)
    na_nonzero = na[na > 0]
    print("\n[NaN check] NaN 있는 컬럼 수:", len(na_nonzero))
    if len(na_nonzero) > 0:
        print(na_nonzero.head(top_na_cols))
    else:
        print("✅ NaN 없음")

    # 중복
    dup_cnt = int(features_df.duplicated().sum())
    print("\n[Duplicate rows] features duplicated:", dup_cnt)

    # 0/1 체크
    bad_cols = []
    for c in features_df.columns:
        vals = set(features_df[c].dropna().unique().tolist())
        if not vals.issubset({0, 1}):
            bad_cols.append((c, list(vals)[:10]))
    if bad_cols:
        print("\n⚠️ [0/1 check] 0/1 이외 값이 있는 컬럼 발견:")
        for c, vals in bad_cols[:20]:
            print(" -", c, ":", vals)
    else:
        print("\n✅ [0/1 check] 모든 feature가 0/1")

    # meta_df도 같이 보면
    if meta_df is not None:
        print("\nmeta_df shape:", meta_df.shape)
        print("meta_df columns:", list(meta_df.columns))
        print("meta_df head:")
        print(meta_df.head(3))

        if len(features_df) != len(meta_df):
            print("\n⚠️ features_df와 meta_df 행 수가 다릅니다! (정렬/필터링 로직 확인 필요)")
        else:
            print("\n✅ features_df와 meta_df 행 수 일치")

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
