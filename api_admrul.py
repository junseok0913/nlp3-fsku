import os
import json
import sys
from typing import Any, Dict, List, Optional
import requests
import re

# ===== 사용자 설정 =====
ADM_RULE_QUERY = "정보보호 및 개인정보보호 관리체계 인증 등에 관한 고시"  # 예시: 검색에 사용할 키워드(정확 명칭이 아니어도 됨)
OC = os.getenv("LAW_OC", "junseok0913")      # 국가법령정보 공동활용에 신청된 본인 OC(이메일 ID 부분)

# ===== 상수 =====
SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"
SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"
TARGET = "admrul"  # 행정규칙

# ===== 유틸 =====
def _collect_items(obj: Any, bag: List[Dict]):
    """lawSearch JSON 응답에서 행정규칙 항목들을 안전하게 수집합니다."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            # 일부 응답에서 'admrul' 키 아래 리스트가 오고,
            # 다른 경우엔 바로 필드들이 섞여 있을 수 있습니다.
            if k.lower() in ("admrul", "admruls"):
                if isinstance(v, list):
                    bag.extend([it for it in v if isinstance(it, dict)])
                elif isinstance(v, dict):
                    bag.append(v)
            else:
                _collect_items(v, bag)
    elif isinstance(obj, list):
        for x in obj:
            _collect_items(x, bag)

def _gv(d: Dict, key: str, default: str = "") -> str:
    """키 이름 변형에 어느 정도 내성을 주는 getter."""
    return d.get(key) or d.get(key.replace("_", "")) or default

def _sanitize_filename(name: str, repl: str = "_") -> str:
    return re.sub(r'[\\/:*?"<>|]+', repl, name).strip()

# ===== 핵심 로직 =====
def search_admrul_candidates(
    oc: str,
    query: str,
    *,
    nw: int = 1,              # 1: 현행, 2: 연혁
    display: int = 100,       # 최대 100
    page: int = 1,
    org: Optional[str] = None,  # 소관부처 코드(선택)
    knd: Optional[str] = None,  # 1=훈령/2=예규/3=고시/4=공고/5=지침/6=기타
    timeout: int = 20,
) -> List[Dict]:
    """행정규칙을 키워드로 검색하여 후보 목록(JSON 원소 dict들)을 반환."""
    params = {
        "OC": oc,
        "target": TARGET,
        "type": "JSON",
        "query": query,
        "nw": nw,
        "display": display,
        "page": page,
    }
    if org:
        params["org"] = org
    if knd:
        params["knd"] = knd

    r = requests.get(SEARCH_URL, params=params, timeout=timeout)
    r.raise_for_status()
    r.encoding = "utf-8"
    data = r.json()

    items: List[Dict] = []
    _collect_items(data, items)

    # 혹시 '행정규칙명'이 없는 잡음 제거
    items = [it for it in items if _gv(it, "행정규칙명")]
    return items

def pick_best_admrul(items: List[Dict], query: str) -> Dict:
    """정확 일치 > 부분 일치 > 첫 항목 순으로 하나 선택."""
    if not items:
        raise RuntimeError("검색 결과가 없습니다. OC 승인/허용 IP/도메인을 확인해 보세요.")

    exact = [it for it in items if _gv(it, "행정규칙명") == query]
    if exact:
        return exact[0]

    part = [it for it in items if query in _gv(it, "행정규칙명")]
    if part:
        return part[0]

    return items[0]

def fetch_admrul_json(
    oc: str,
    *,
    id_value: Optional[str] = None,   # 행정규칙 일련번호(ID)
    lid_value: Optional[str] = None,  # 행정규칙 ID(LID)
    lm_value: Optional[str] = None,   # 행정규칙명(LM) - 정확 명칭 필요
    timeout: int = 30,
) -> Dict:
    """행정규칙 본문(JSON)을 조회. ID/LID/LM 중 하나만 넘기면 됩니다."""
    if not (id_value or lid_value or lm_value):
        raise ValueError("id_value, lid_value, lm_value 중 하나는 반드시 필요합니다.")

    params = {
        "OC": oc,
        "target": TARGET,
        "type": "JSON",
    }
    if id_value:
        params["ID"] = id_value
    if lid_value:
        params["LID"] = lid_value
    if lm_value:
        params["LM"] = lm_value  # LM은 정확한 행정규칙명을 요구

    r = requests.get(SERVICE_URL, params=params, timeout=timeout, headers={"Accept": "application/json"})
    r.raise_for_status()
    r.encoding = "utf-8"
    return r.json()

# ===== 실행부 =====
if __name__ == "__main__":
    if not OC:
        print("환경변수 LAW_OC 또는 코드의 OC 변수에 본인의 OC(이메일 ID 부분)를 설정하세요.", file=sys.stderr)
        sys.exit(1)

    try:
        # 1) 키워드로 후보 검색
        candidates = search_admrul_candidates(OC, ADM_RULE_QUERY, nw=1, display=100)

        # 2) 최적 후보 선택
        best = pick_best_admrul(candidates, ADM_RULE_QUERY)

        # 검색 결과에서 식별자 뽑기
        #   - '행정규칙 일련번호'가 본문 조회의 ID 파라미터
        #   - '행정규칙ID'는 본문 조회의 LID 파라미터
        admrul_name = _gv(best, "행정규칙명", "행정규칙")
        admrul_id = _gv(best, "일련번호") or _gv(best, "행정규칙일련번호")  # ID
        admrul_lid = _gv(best, "행정규칙ID")                               # LID

        if not (admrul_id or admrul_lid):
            raise RuntimeError("검색 결과에서 ID/LID를 찾지 못했습니다.")

        # 3) 본문 조회 (ID가 있으면 ID 우선)
        data = fetch_admrul_json(OC, id_value=admrul_id, lid_value=None if admrul_id else admrul_lid)

        # 4) 보기 좋게 출력 및 저장
        pretty = json.dumps(data, ensure_ascii=False, indent=2)
        print(pretty)

        filename = f"data/raw/{_sanitize_filename(admrul_name)}.json"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(pretty)
        print(f"\n저장 완료: {filename}")

    except requests.HTTPError as e:
        print(f"HTTP 오류: {e} / 응답: {getattr(e.response, 'text', '')[:500]}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"실패: {e}", file=sys.stderr)
        sys.exit(3)
