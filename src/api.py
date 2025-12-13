# src/api.py
import requests
import pandas as pd

BASE_URL = "http://openapi.foodsafetykorea.go.kr/api"
API_KEY = "sample"  # 지금은 샘플키 고정

def fetch_food_data(service_id: str, start_idx: int = 1, end_idx: int = 100, data_type: str = "json"):
    """
    returns: (df, total_count)
    """
    url = f"{BASE_URL}/{API_KEY}/{service_id}/{data_type}/{start_idx}/{end_idx}"
    r = requests.get(url, timeout=30)

    text = r.text if r.text is not None else ""
    ctype = (r.headers.get("content-type") or "").lower()

    # 1) HTTP 체크
    if r.status_code != 200:
        raise RuntimeError(f"[HTTP {r.status_code}] {url}\nhead={text[:300]}")

    # 2) JSON 여부 체크(서버가 HTML 줄 때 차단)
    # - 공백 제거
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        raise RuntimeError(
            "JSON이 아닌 응답입니다.\n"
            f"url={url}\ncontent-type={ctype}\nhead={stripped[:300]}"
        )

    # 3) 여기서도 json 파싱 실패할 수 있으니 try로 감싸서 원문을 보여줌
    try:
        data = r.json()
    except Exception as e:
        raise RuntimeError(
            "JSON 파싱 실패(응답이 깨졌거나 중간에 잘렸을 수 있음)\n"
            f"url={url}\ncontent-type={ctype}\nhead={stripped[:300]}"
        ) from e

    if service_id not in data:
        raise RuntimeError(f"응답 JSON에 '{service_id}' 키가 없습니다. keys={list(data.keys())}")

    body = data[service_id]

    # API 결과 코드 확인
    result = body.get("RESULT", {})
    if result and result.get("CODE") not in (None, "INFO-000"):
        raise RuntimeError(f"API 오류: {result}")

    df = pd.DataFrame(body.get("row", []))
    total = int(body.get("total_count", 0) or 0)
    return df, total
