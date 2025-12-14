import os
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env", override=True)

BASE_URL = "https://openapi.foodsafetykorea.go.kr/api"
API_KEY = (os.getenv("FOOD_API_KEY", "") or "").strip().replace("\ufeff", "")

def _request_json(url: str, timeout: int = 30):
    r = requests.get(url, timeout=timeout)
    text = (r.text or "").strip()
    ctype = (r.headers.get("content-type") or "").lower()
    return r.status_code, ctype, text, r

def fetch_food_data(service_id: str, start_idx: int = 1, end_idx: int = 100, data_type: str = "json", use_sample_fallback: bool = True):
    if API_KEY:
        url = f"{BASE_URL}/{API_KEY}/{service_id}/{data_type}/{start_idx}/{end_idx}"
        status, ctype, text, r = _request_json(url)

        if status == 200 and text.startswith("{"):
            data = r.json()
            if service_id in data and "row" in data[service_id]:
                body = data[service_id]
                df = pd.DataFrame(body.get("row", []))
                total = int(body.get("total_count", 0) or 0)
                print("✅ REAL API USED:", url)
                return df, total

        if (not use_sample_fallback):
            raise RuntimeError(f"API 실패(URL={url})\nstatus={status}\nctype={ctype}\nhead={text[:300]}")

    url2 = f"{BASE_URL}/sample/{service_id}/{data_type}/{start_idx}/{end_idx}"
    status2, ctype2, text2, r2 = _request_json(url2)

    if status2 != 200 or (not text2.startswith("{")):
        raise RuntimeError(
            "API가 JSON이 아닌 응답을 줬습니다.\n"
            f"URL={url2}\nstatus={status2}\ncontent-type={ctype2}\nhead={text2[:300]}"
        )

    data2 = r2.json()
    if service_id not in data2:
        raise RuntimeError(f"응답에 {service_id} 키가 없습니다. keys={list(data2.keys())[:10]}")

    body2 = data2[service_id]
    df2 = pd.DataFrame(body2.get("row", []))
    total2 = int(body2.get("total_count", 0) or 0)
    print("⚠️ SAMPLE API USED:", url2)
    return df2, total2
