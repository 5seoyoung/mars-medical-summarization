# =========================
# submit.ipynb / submit.py  (Team. MediX)
# =========================
# 본선 규칙 준수용 최종 코드 (10/16 공지 반영 본)
# - 입력: ./test 디렉토리 내 6개 CSV (discharge_summary.csv / 퇴원기록.csv 미사용)
# - 처리: admission.csv의 모든 환자번호에 대해 일괄 생성
# - 산출물: result.csv(["환자번호","summary"]), runtime.txt(총 소요초, 소수 둘째)
# - 문서 구조: ★필수 5슬롯(이름·순서 고정)로 요약 텍스트 생성
# - 데이터 경계: 입원≤t≤퇴원 구간만 사용 (퇴원 서식/문서 입력 금지)
# - LLM OFF (규칙/정규식 기반) → 결정적 출력 & 재현성
# - MeSH 2025: 있으면 탐지/정규화 강화, 없으면 자동 폴백
# - 제출 요건: 실행 종료 시 result.csv / runtime.txt를 제출 버킷에 업로드 시도
#   s3://snubh-2025-datathon-finals/submissions/<team_name>
# =========================

# 0) Execution timer (whole run) ----------------------------------------------
import time as _time
T0 = _time.time()  # 전체 실행시간 측정 시작 (runtime.txt 기록용)

# 1) Imports ------------------------------------------------------------------
from pathlib import Path
import os, subprocess, sys, re
import pandas as pd
import numpy as np

# 1.1) S3 업로드 설정 ----------------------------------------------------------
TEAM_NAME   = os.getenv("TEAM_NAME", "MediX")
S3_BUCKET   = os.getenv("S3_BUCKET", "snubh-2025-datathon-finals")
S3_PREFIX   = f"submissions/{TEAM_NAME}"  # s3://snubh-2025-datathon-finals/submissions/MediX
UPLOAD_TO_S3= os.getenv("UPLOAD_TO_S3", "true").lower() not in ("0","false","no")
S3_STRICT   = os.getenv("S3_STRICT", "false").lower() in ("1","true","yes")  # True면 업로드 실패 시 예외
S3_ACL      = os.getenv("S3_ACL", "").strip()  # 예: "bucket-owner-full-control"

def _s3_uri(filename: str) -> str:
    return f"s3://{S3_BUCKET}/{S3_PREFIX}/{filename}"

def _upload_via_awscli(local_path: str, s3_uri: str) -> None:
    """awscli 업로드. 필요 시 --acl 옵션 부여."""
    cmd = ["aws", "s3", "cp", local_path, s3_uri, "--only-show-errors"]
    if S3_ACL:
        cmd += ["--acl", S3_ACL]
    subprocess.run(cmd, check=True)

def _upload_via_boto3(local_path: str, s3_uri: str) -> None:
    """boto3 백업 경로. 필요 시 ExtraArgs에 ACL 부여."""
    import boto3
    from botocore.config import Config
    s3 = boto3.client("s3", config=Config(retries={"max_attempts": 3, "mode": "standard"}))
    key = f"{S3_PREFIX}/{os.path.basename(local_path)}"
    extra = {}
    if S3_ACL:
        extra["ExtraArgs"] = {"ACL": S3_ACL}
    if extra:
        s3.upload_file(local_path, S3_BUCKET, key, **extra)
    else:
        s3.upload_file(local_path, S3_BUCKET, key)

def upload_to_s3_with_retry(local_path: str, attempts: int = 2):
    """awscli → boto3 순차 시도, 지정 횟수 재시도. 실패 시 S3_STRICT에 따라 경고 또는 예외."""
    s3_uri = _s3_uri(os.path.basename(local_path))
    last_err = None
    for _ in range(attempts):
        try:
            _upload_via_awscli(local_path, s3_uri)
            if os.getenv("DEBUG"): print(f"[S3] Uploaded via awscli -> {s3_uri}")
            return
        except Exception as e_cli:
            last_err = e_cli
            try:
                _upload_via_boto3(local_path, s3_uri)
                if os.getenv("DEBUG"): print(f"[S3] Uploaded via boto3 -> {s3_uri}")
                return
            except Exception as e_boto:
                last_err = (e_cli, e_boto)
    # 모두 실패
    msg = (f"[S3][WARN] 업로드 실패 → 로컬 유지: {local_path}\n"
           f" - 대상: {s3_uri}\n"
           f" - 원인(추정): 권한 미부여/리전 불일치/버킷 정책(ACL) 요구\n"
           f" - 권한 후 재시도: aws s3 cp {local_path} {s3_uri}"
           + (f" --acl {S3_ACL}" if S3_ACL else ""))
    print(msg, file=sys.stderr)
    if S3_STRICT:
        raise RuntimeError(f"S3 업로드 실패(details={last_err})")

# 2) Settings & Utilities -----------------------------------------------------
TEST_DIR   = Path("test")   # ★ 공지: test 디렉토리 고정
SAMPLE_DIR = Path("sample")
DEBUG = os.getenv("DEBUG", "false").lower() in ("1","true","yes")

# LLM(옵션) — 기본 OFF: 재현성·결정성 유지
USE_LLM = False
MODEL_PATH = "models/Llama-3.1-8B-Instruct"
MAX_TOTAL_CHARS = 1200  # 출력 길이 상한 (가독/일관성)

# 가이드 준수: 향후 계획/외래 안내/일반 권고는 작성 제외
INCLUDE_FUTURE_PLAN = False   # ★ 반드시 False (가이드 명시사항 준수)
# (선택) 과별 힌트 문구(부족 키워드 암시) — 공식 제출물은 편집성 문장 배제 권장
ADD_SPECIALTY_HINTS_OFFICIAL = False

# (선택) MeSH 2025 자산 1회 자동 빌드 트리거 (제출 안정성 위해 기본 False)
# - requested-data/ 경로에 MeSH XML이 존재할 때만 시도
BUILD_MESH_ASSETS_ONCE = False

def read_csv_safe(p: Path, **kw):
    """안전 CSV 로더: UTF-8/UTF-8-SIG만 시도."""
    if not p.exists():
        raise FileNotFoundError(p)
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return pd.read_csv(p, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"Failed to decode {p} with utf-8 / utf-8-sig")

def to_dt(s):
    """문자열→Timestamp (파싱 실패는 NaT로)"""
    return pd.to_datetime(s, errors="coerce")

def s(x):
    """NaN 안전 문자열화"""
    if pd.isna(x): return ""
    return str(x).strip()

def join_lines(lines, max_chars=1200):
    """라인 결합 + 전체 길이 제한(가독성/일관성 보호)"""
    out, size = [], 0
    for ln in lines:
        ln = re.sub(r"\s+", " ", s(ln)).strip()
        if not ln: continue
        if size + len(ln) + 1 > max_chars: break
        out.append(ln); size += len(ln) + 1
    return "\n".join(out)

def post_kor(text: str) -> str:
    """한국어 판독성 경량 후처리"""
    text = re.sub(r"(^|[^\w])#\s*\.?", r"\1• ", text)
    text = text.replace("s/p", "시행 후").replace("PMHx", "과거력").replace("F/U", "경과 관찰")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

# 3) Load data ----------------------------------------------------------------
# (공지 준수: discharge_summary.csv / 퇴원기록.csv는 입력 금지, 경로는 ./test 고정)
admission          = read_csv_safe(TEST_DIR/"admission.csv",          low_memory=False)
diagnosis          = read_csv_safe(TEST_DIR/"diagnosis.csv",          low_memory=False)
surgery_anesthesia = read_csv_safe(TEST_DIR/"surgery_anesthesia.csv", low_memory=False)
nursing_note       = read_csv_safe(TEST_DIR/"nursing_note.csv",       low_memory=False)

# 중요 패치) medical_note에 섞여 들어온 '퇴원' 서식 라인을 런타임에서 배제
_medical_note_raw  = read_csv_safe(TEST_DIR/"medical_note.csv",       low_memory=False)
if "서식명" in _medical_note_raw.columns:
    _drop_mask = _medical_note_raw["서식명"].astype(str).str.contains("퇴원", na=False)
    _removed_rows = int(_drop_mask.sum())
    medical_note = _medical_note_raw.loc[~_drop_mask].copy()
    if DEBUG and _removed_rows:
        print(f"[WARN] medical_note에서 '퇴원' 서식명 {_removed_rows}행 제거")
else:
    medical_note = _medical_note_raw.copy()

chief_complaint    = read_csv_safe(TEST_DIR/"chief_complaint.csv",    low_memory=False)

# Stay & time filter ----------------------------------------------------------
def get_stay(pid: int):
    """환자별 입원~퇴원 구간 반환."""
    rows = admission.loc[admission["환자번호"] == pid]
    if rows.empty:
        raise ValueError(f"환자번호 {pid} 없음")
    row = rows.iloc[0]
    t_in  = to_dt(row["입원일자"])
    t_out = to_dt(row["퇴원일자"])
    if pd.isna(t_in) or pd.isna(t_out):
        raise ValueError(f"환자번호 {pid} 입원/퇴원일 결측")
    return t_in, t_out

def within_stay(df, pid, id_col, date_col, t_in, t_out):
    """입원≤date≤퇴원 구간 필터 + 정렬."""
    dd = df.loc[df[id_col] == pid].copy()
    dd[date_col] = to_dt(dd[date_col])
    ok = dd[date_col].notna() & (dd[date_col] >= t_in) & (dd[date_col] <= t_out)
    return dd.loc[ok].sort_values(date_col)

# 4) Section builders ----------------------------------------------------------
def _get_dept(row) -> str:
    """입원과 컬럼명(공백 유무)에 대한 호환 처리"""
    return s(row.get("입원시 진료과") or row.get("입원 시 진료과"))

def make_header(pid, t_in, t_out):
    """인적/입원정보 헤더"""
    row  = admission.loc[admission["환자번호"] == pid].iloc[0]
    sex  = s(row.get("성별"))
    age  = s(row.get("수진 당시 나이"))
    dept = _get_dept(row)
    days = s(row.get("재원 일수(응급실 미포함)"))
    return f"{sex}, {age}, 입원기간 {t_in.date()}~{t_out.date()}({days}일), 입원과 {dept}"

def make_cc(pid, t_in, t_out, topk=3):
    """주호소(CC) 요약"""
    df = within_stay(chief_complaint, pid, "환자번호", "작성일자", t_in, t_out)
    vals = df["원내 CC명"].dropna().astype(str).unique().tolist()
    return ", ".join(vals[:topk])

def make_dx(pid, t_in, t_out, k=6):
    """주요 진단: 최신 k개 (코드:명칭)"""
    df = within_stay(diagnosis, pid, "환자번호", "진단일자", t_in, t_out)
    df = df.sort_values("진단일자", ascending=False).head(k)
    lines = [f"{s(r['KCD 코드'])}: {s(r['KCD 한글명'])}" for _, r in df.iterrows()]
    return "; ".join(lines)

def make_op(pid, t_in, t_out, k=3):
    """수술/마취 요약 (있으면)"""
    df = within_stay(surgery_anesthesia, pid, "환자번호", "수술 일자", t_in, t_out)
    if df.empty:
        return ""
    out = []
    for _, r in df.sort_values("수술 일자").head(k).iterrows():
        bits = [s(r.get("ICD9CM 명"))]
        asa  = s(r.get("[마취전 상태평가] ASA class"))
        anes = s(r.get("[수술기록] 마취종류"))
        bld  = s(r.get("[수술실 퇴실전] 출혈정도"))
        meta = ", ".join([v for v in [f"ASA {asa}" if asa else "", anes, bld] if v])
        if meta:
            bits.append(f"({meta})")
        out.append(" ".join(bits))
    return "; ".join(out)

# 경과 추출 키워드
KEY_NOTE_FIELDS = re.compile(r"(Assessment|Impression|Plan|진단|경과|요약|소견)", re.I)

def make_med_course(pid, t_in, t_out, max_chars=800):
    """입원경과 요약"""
    # Medical notes
    m = within_stay(medical_note, pid, "환자번호", "서식 작성일자", t_in, t_out)
    m_lines = []
    for _, r in m.iterrows():
        name = s(r.get("서식 항목명"))
        val  = s(r.get("항목별 서식 값"))
        if not val:
            continue
        if KEY_NOTE_FIELDS.search(name) or KEY_NOTE_FIELDS.search(val):
            m_lines.append(f"{name}: {val}")
    # Nursing notes
    n = within_stay(nursing_note, pid, "환자번호", "간호기록 작성일자", t_in, t_out)
    n_key = re.compile(r"(내원 주증상|입원동기|의식|활력징후|배변|섭취|통증|산소|증상|간호)", re.I)
    n_lines = []
    for _, r in n.iterrows():
        key = s(r.get("항목명"))
        val = s(r.get("항목값"))
        if n_key.search(key) and val:
            n_lines.append(f"{key}: {val}")
    return join_lines(m_lines + n_lines, max_chars=max_chars)

# 5) 검사결과 추출(기본 정규식/라벨) -------------------------------------------
_NUM_PAT = re.compile(
    r"(?P<name>(Hb|Hemoglobin|헤모글로빈|WBC|Cr|Creatinine|크레아티닌|BUN|Na|K|칼륨|Sodium|ALT|AST|CRP|Troponin|EF|eGFR|Platelet|혈소판)[^:\n]{0,30})"
    r"[:\s]*"
    r"(?P<val>[-+]?\d+(?:\.\d+)?)\s*(?P<unit>%|mg/dL|g/dL|U/L|mmol/L|mEq/L|×?10\^?3/?u?L|ng/L|mL/min/1\.73m²)?",
    re.I
)

IMG_HINT  = re.compile(r"(CT|MRI|X[- ]?ray|Ultrasound|Echo|Echocardiography|angiography|내시경|초음파|흉부단순|CAG|PCI)", re.I)
FUNC_HINT = re.compile(r"(PFT|ABGA|EKG|ECG|EEG|Holter|TTE|TTE/TEE|EF)", re.I)
PATH_HINT = re.compile(r"(biopsy|조직검사|세포검사|병리|pathology)", re.I)

_SYNONYM_CANON = [
    (re.compile(r"creatinine|[^a-z]cr[^a-z]|\bcr\b", re.I), "Creatinine (Cr)"),
    (re.compile(r"hemoglobin|hb|헤모글로빈", re.I),          "Hemoglobin (Hb)"),
    (re.compile(r"\bwbc\b", re.I),                          "WBC"),
    (re.compile(r"platelet|혈소판", re.I),                  "Platelet"),
    (re.compile(r"egfr", re.I),                             "eGFR"),
    (re.compile(r"troponin", re.I),                         "Troponin"),
    (re.compile(r"\bna\b|sodium", re.I),                    "Sodium (Na)"),
    (re.compile(r"\bk\b|칼륨", re.I),                       "Potassium (K)"),
]

def _canonize(name: str) -> str:
    """검사 항목명 정규화 — MeSH 매핑이 있다면 뒤에서 덮어씀"""
    for pat, c in _SYNONYM_CANON:
        if pat.search(name):
            return c
    return re.sub(r"\s+", " ", name).strip(" :")

def _scan_note_rows(df, date_col, cols):
    """노트 테이블을 (날짜, 텍스트) 묶음으로 평탄화"""
    out = []
    for _, r in df.iterrows():
        d  = to_dt(r.get(date_col))
        txt = " ".join(s(r.get(c)) for c in cols)
        out.append((d, txt))
    return out

# 5.1) === MeSH 2025 통합 (있으면 강화, 없으면 폴백) ==========================
def _maybe_build_mesh_assets():
    """requested-data/에 MeSH XML이 있고 BUILD_MESH_ASSETS_ONCE=True면 assets/* 생성"""
    if not BUILD_MESH_ASSETS_ONCE:
        return False
    try:
        import xml.etree.ElementTree as ET, gzip, json
        BASE = Path("requested-data")
        ASSETS_DIR = Path("assets"); ASSETS_DIR.mkdir(parents=True, exist_ok=True)

        def _open(p: Path):
            return gzip.open(p, "rb") if p.suffix == ".gz" else open(p, "rb")

        def _find(keys):
            for p in BASE.rglob("*"):
                name = p.name.lower()
                if all(k in name for k in keys):
                    return p
            return None

        D = _find(["descriptor", ".xml"]) or _find(["d2025", ".xml"])
        S = _find(["supplement", ".xml"]) or _find(["c2025", ".xml"])
        if not D and not S:
            return False

        def parse_desc(p):
            out=[]
            for ev,elem in ET.iterparse(_open(p), events=("end",)):
                if elem.tag.lower().endswith("descriptorrecord"):
                    def T(path):
                        n = elem.find(path)
                        return (n.text or "").strip() if n is not None else ""
                    ui   = T("./DescriptorUI")
                    name = T("./DescriptorName/String")
                    terms=[]
                    for c in elem.findall("./ConceptList/Concept"):
                        for t in c.findall("./TermList/Term/String"):
                            if t.text: terms.append(t.text.strip())
                    terms = sorted(set(x for x in terms if x and x.lower()!=name.lower()))
                    out.append({"ui":ui,"name":name,"entry_terms":terms})
                    elem.clear()
            return out

        def parse_scr(p):
            out=[]
            for ev,elem in ET.iterparse(_open(p), events=("end",)):
                if elem.tag.lower().endswith("supplementaryconceptrecord"):
                    def T(path):
                        n = elem.find(path)
                        return (n.text or "").strip() if n is not None else ""
                    ui   = T("./SupplementalRecordUI")
                    name = T("./SupplementalRecordName/String")
                    terms=[]
                    for t in elem.findall("./ConceptList/Concept/TermList/Term/String"):
                        if t.text: terms.append(t.text.strip())
                    terms = sorted(set(x for x in terms if x and x.lower()!=name.lower()))
                    out.append({"ui":ui,"name":name,"entry_terms":terms})
                    elem.clear()
            return out

        recs = []
        if D: recs += parse_desc(D)
        if S: recs += parse_scr(S)
        if not recs:
            return False

        import json, re
        KEY_IMAGING  = re.compile(r"(tomograph|magnetic resonance|mri|ultra(?:sound|sonograph)|echocardiograph|endoscop|angiograph|radiograph|x-?ray|scintigraph)", re.I)
        KEY_FUNCTION = re.compile(r"(electrocardiograph|ecg|ekg|electroencephalograph|eeg|pulmonary function|spirometr|abg|holter|pft|\bef\b)", re.I)
        KEY_PATH     = re.compile(r"(biopsy|histolog|cytolog|patholog)", re.I)
        LAB_SEEDS = [
            "Creatinine", "Hemoglobin", "C-Reactive Protein", "Sodium", "Potassium",
            "Platelet Count", "Leukocyte Count", "Erythrocyte Count",
            "Troponin", "Ejection Fraction", "Glomerular Filtration Rate"
        ]
        img,function,path,lab_alias = set(),set(),set(),set()
        for r in recs:
            terms = [r["name"], *r["entry_terms"]]
            blob  = " | ".join(terms)
            if KEY_IMAGING.search(blob):  img.update(terms)
            if KEY_FUNCTION.search(blob): function.update(terms)
            if KEY_PATH.search(blob):     path.update(terms)
            for seed in LAB_SEEDS:
                for t in terms:
                    if seed.lower() in (t or "").lower():
                        lab_alias.add((t.strip(), seed))

        def norm(S):
            return sorted(set(re.sub(r"\s+"," ",x).strip() for x in S if x and len(x)<=80))[:5000]

        with open(ASSETS_DIR/"mesh_synonyms.json","w",encoding="utf-8") as f:
            json.dump({"imaging":norm(img), "function":norm(function), "pathology":norm(path)},
                      f, ensure_ascii=False, indent=2)
        with open(ASSETS_DIR/"mesh_lab_alias.csv","w",encoding="utf-8") as f:
            f.write("term,canon\n")
            for term,canon in sorted(set(lab_alias)):
                f.write(f"{term},{canon}\n")
        return True
    except Exception:
        return False

def _compile_hint_from_mesh(json_path="assets/mesh_synonyms.json"):
    """MeSH 기반 카테고리 정규식 로더 (없으면 기본 힌트 폴백)"""
    try:
        import json
        with open(json_path,"r",encoding="utf-8") as f:
            obj = json.load(f)
        def pat(words):
            words = [w for w in words if w][:1000]
            esc = [re.escape(w) for w in words]
            return re.compile("|".join(esc), re.I) if esc else re.compile(r"$^")
        return {
            "IMG_HINT":  pat(obj.get("imaging", [])),
            "FUNC_HINT": pat(obj.get("function", [])),
            "PATH_HINT": pat(obj.get("pathology", [])),
        }
    except Exception:
        return {"IMG_HINT": IMG_HINT, "FUNC_HINT": FUNC_HINT, "PATH_HINT": PATH_HINT}

def _load_mesh_lab_alias(csv_path="assets/mesh_lab_alias.csv"):
    """MeSH 기반 Lab 라벨 정규화 사전 (없으면 빈 dict → 기본 사전만 사용)"""
    mp = {}
    try:
        with open(csv_path,"r",encoding="utf-8") as f:
            header = next(f, "")
            for line in f:
                if not line.strip(): continue
                term, canon = line.rstrip("\n").split(",", 1)
                mp[term.lower()] = canon
    except Exception:
        pass
    return mp

# MeSH 자산 자동 빌드(옵션) & 로드 -------------------------------------------
_ = _maybe_build_mesh_assets()
_hints = _compile_hint_from_mesh()
IMG_HINT, FUNC_HINT, PATH_HINT = _hints["IMG_HINT"], _hints["FUNC_HINT"], _hints["PATH_HINT"]
_MESH_LAB_ALIAS = _load_mesh_lab_alias()

# MeSH alias 우선 적용하도록 _canonize 재정의
def _canonize(name: str) -> str:
    nm = re.sub(r"\s+", " ", name).strip(" :")
    if nm.lower() in _MESH_LAB_ALIAS:
        return _MESH_LAB_ALIAS[nm.lower()]
    for pat, c in _SYNONYM_CANON:
        if pat.search(nm):
            return c
    return nm

# 6) 검사결과 추출 -------------------------------------------------------------
def extract_key_results(pid, t_in, t_out, max_lines=12) -> str:
    """
    검사결과(★): Lab/Imaging/Function/Pathology 4블록
    - Lab: 수치/단위 정규식 파싱 + 동의어 정규화 + 동일 항목 최신값 최대 2개 선택
    - Imaging/Function/Pathology: (MeSH 통합) 힌트 정규식 기반 탐지, 첫 문장 발췌 + 날짜
    """
    m = within_stay(medical_note, pid, "환자번호", "서식 작성일자", t_in, t_out)
    n = within_stay(nursing_note, pid, "환자번호", "간호기록 작성일자", t_in, t_out)
    rows = (
        _scan_note_rows(m, "서식 작성일자", ["서식 항목명","항목별 서식 값"]) +
        _scan_note_rows(n, "간호기록 작성일자", ["항목명","항목값"])
    )

    labs_map = {}   # canon -> list[(date, "[d] name: valunit")]
    imgs, funcs, paths = [], [], []

    for d, txt in rows:
        dtag = f"[{d.date()}]" if pd.notna(d) else ""
        # Lab 수치
        for mobj in _NUM_PAT.finditer(txt):
            raw_name = re.sub(r"\s+", " ", mobj.group("name")).strip(" :")
            canon    = _canonize(raw_name)
            val      = mobj.group("val")
            unit     = mobj.group("unit") or ""
            item_txt = f"{dtag} {canon}: {val}{unit}".strip()
            labs_map.setdefault(canon, []).append((d, item_txt))
        # Imaging / Function / Pathology (첫 문장)
        snippet = re.split(r"[.\n]", txt)[0].strip()
        if IMG_HINT.search(txt):   imgs.append(f"{dtag} {snippet}")
        if FUNC_HINT.search(txt):  funcs.append(f"{dtag} {snippet}")
        if PATH_HINT.search(txt):  paths.append(f"{dtag} {snippet}")

    # 동일 항목 최신값 1~2개
    labs = []
    for canon, items in labs_map.items():
        items.sort(key=lambda x: (pd.Timestamp.min if pd.isna(x[0]) else x[0]), reverse=True)
        pick = [t for _, t in items[:2]]
        labs.extend(pick)

    def dedup_keep_order(seq):
        seen, out = set(), []
        for x in seq:
            key = re.sub(r"\s+", " ", x).strip().lower()
            if key and key not in seen:
                seen.add(key); out.append(x)
        return out

    labs  = dedup_keep_order(labs)[:max(4, max_lines//2)]
    imgs  = dedup_keep_order(imgs)[:max_lines//3]
    funcs = dedup_keep_order(funcs)[:max_lines//4]
    paths = dedup_keep_order(paths)[:max_lines//4]

    blocks = []
    if labs:  blocks += ["- Labs:",     *[f"  • {x}" for x in labs[:6]]]
    if imgs:  blocks += ["- Imaging:",  *[(f"  • {x}") for x in imgs[:4]]]
    if funcs: blocks += ["- Function:", *[(f"  • {x}") for x in funcs[:4]]]
    if paths: blocks += ["- Pathology:",*[(f"  • {x}") for x in paths[:4]]]
    return "\n".join(blocks[:max_lines]) if blocks else "해당 항목 기재 없음"

# 7) 과별 힌트(기본 OFF) -------------------------------------------------------
def guess_specialty(dept: str) -> str:
    d = (dept or "").strip()
    if "순환" in d or "심장" in d: return "cardio"
    if "신경" in d:               return "neuro"
    if "신장" in d:               return "nephro"
    if "소화" in d or "위장" in d: return "gi"
    return "other"

SPECIALTY_RULES = {
    "cardio": {"must": ["EF", "ECG", "CAG", "PCI", "troponin"], "hint": "심혈관 핵심(EF/ECG/CAG/PCI, troponin) 반영"},
    "neuro":  {"must": ["수술", "병변", "신경학적", "출혈", "감염"], "hint": "수술/병변/신경학적 변화/합병증"},
    "nephro": {"must": ["Cr", "eGFR", "투석", "biopsy", "proteinuria"], "hint": "신기능 추이/투석 여부/병리"},
    "gi":     {"must": ["내시경", "조직", "출혈", "천공", "간"], "hint": "내시경·조직 및 합병증"},
}

def enforce_specialty_points(dept: str, text: str) -> str:
    sp = guess_specialty(dept)
    rule = SPECIALTY_RULES.get(sp)
    if not rule: 
        return text
    miss = [k for k in rule["must"] if k.lower() not in text.lower()]
    if not miss:
        return text
    return (text + f"\n(참고: {rule['hint']} — 원문상 명시된 항목만 기재)").strip()

# 8) 5-슬롯 템플릿/생성 ---------------------------------------------------------
REQUIRED_SLOTS = [
    "admission_reason_history",
    "hospital_course",
    "outcome",
    "key_results",
    "patient_summary"
]

TEMPLATE_5 = """[퇴원요약]
입원사유·병력요약: {admission_reason_history}
입원경과: {hospital_course}
입원결과: {outcome}
검사결과: 
{key_results}
Patient Summary:
{patient_summary}
""".strip()

def _fill_required_5(sections: dict) -> dict:
    filled = dict(sections)
    for k in REQUIRED_SLOTS:
        if not filled.get(k):
            filled[k] = "해당 항목 기재 없음"
    return filled

def format_with_template_5(sections: dict) -> str:
    safe = _fill_required_5(sections)
    text = TEMPLATE_5.format(**safe).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return post_kor(text)

def make_admission_reason_history(pid, t_in, t_out):
    """입원사유·병력요약(★): CC + 주요 진단 + (가능 시) 과거력"""
    cc  = make_cc(pid, t_in, t_out)
    dx  = make_dx(pid, t_in, t_out)
    m   = within_stay(medical_note, pid, "환자번호", "서식 작성일자", t_in, t_out)
    hx_lines = []
    for _, r in m.iterrows():
        name, val = s(r.get("서식 항목명")), s(r.get("항목별 서식 값"))
        if re.search(r"(PHx|과거력|PMH|history)", name, re.I) or re.search(r"(과거력|병력|PMH)", val):
            hx_lines.append(re.split(r"[\n;]", val)[0])
    hx = "; ".join(dict.fromkeys([re.sub(r"\s+"," ",x) for x in hx_lines]))[:200]
    parts = [f"주호소: {cc}" if cc else "", f"주요 진단: {dx}" if dx else "", f"과거력: {hx}" if hx else ""]
    return "; ".join([p for p in parts if p]).strip()

def make_outcome(pid, t_in, t_out, include_plan: bool = INCLUDE_FUTURE_PLAN):
    """입원결과(★): 퇴원 시 상태 한 줄. 향후 계획은 미기재."""
    m = within_stay(medical_note, pid, "환자번호", "서식 작성일자", t_in, t_out).tail(50)
    candidates = []
    for _, r in m.iterrows():
        val = s(r.get("항목별 서식 값"))
        if re.search(r"(퇴원|상태|호전|유지|악화)", val):
            candidates.append(re.split(r"[\n]", val)[0].strip())
    status = candidates[-1] if candidates else "퇴원 시 전신상태 안정으로 판단."
    return status

def make_patient_summary(pid, t_in, t_out, sections):
    """Patient Summary(★): 헤더/사유·병력/경과/결과 기반 5~7문장"""
    row  = admission.loc[admission["환자번호"] == pid].iloc[0]
    dept = _get_dept(row)
    head = make_header(pid, t_in, t_out)
    lines = [
        head,
        sections.get("admission_reason_history","")[:220],
        re.sub(r"\s+", " ", sections.get("hospital_course",""))[:240],
        "Outcome: " + re.split(r"\n", sections.get("outcome",""))[0][:200],
        "Key results included above.",
    ]
    text = "\n".join([x for x in lines if x]).strip()
    if ADD_SPECIALTY_HINTS_OFFICIAL:
        text = enforce_specialty_points(dept, text)
    return text

def extract_sections_5(pid: int) -> dict:
    """환자 단위로 5슬롯 구성 요소를 추출/가공해 섹션 dict 반환."""
    t_in, t_out = get_stay(pid)
    row   = admission.loc[admission["환자번호"]==pid].iloc[0]
    dept  = _get_dept(row)

    arhx   = make_admission_reason_history(pid, t_in, t_out)
    course = make_med_course(pid, t_in, t_out, max_chars=800)
    outcome= make_outcome(pid, t_in, t_out, include_plan=INCLUDE_FUTURE_PLAN)
    kres   = extract_key_results(pid, t_in, t_out, max_lines=12)

    if ADD_SPECIALTY_HINTS_OFFICIAL:
        arhx   = enforce_specialty_points(dept, arhx)
        course = enforce_specialty_points(dept, course)
        outcome= enforce_specialty_points(dept, outcome)
        kres   = enforce_specialty_points(dept, kres)

    sections = {
        "admission_reason_history": arhx or "해당 항목 기재 없음",
        "hospital_course":          course or "해당 항목 기재 없음",
        "outcome":                  outcome or "해당 항목 기재 없음",
        "key_results":              kres or "해당 항목 기재 없음",
        "patient_summary":          make_patient_summary(pid, t_in, t_out, {
                                      "admission_reason_history": arhx,
                                      "hospital_course": course,
                                      "outcome": outcome,
                                  }) or "해당 항목 기재 없음",
    }
    return sections

def build_summary(pid: int, max_total_chars=MAX_TOTAL_CHARS) -> str:
    """환자별 최종 텍스트 생성: 5슬롯 템플릿 + 길이 상한"""
    sections = extract_sections_5(pid)
    text = format_with_template_5(sections)
    return text[:max_total_chars]

# 9) I/O ----------------------------------------------------------------------
RESULT_COLS = ["환자번호", "summary"]  # ★ 공지: 컬럼명 summary 고정

def run_submit(patient_id, out_csv="result.csv") -> str:
    """단건 제출 실행: 텍스트 생성 → result.csv 1행 기록 → 텍스트 반환"""
    pid  = int(np.int64(patient_id))
    text = build_summary(pid)
    pd.DataFrame([{"환자번호": pid, "summary": text}], columns=RESULT_COLS)\
      .to_csv(out_csv, index=False, encoding="utf-8-sig")
    return text

def run_submit_batch(pids: list[int], out_csv="result.csv"):
    """
    배치 제출 실행(본선 테스트셋 전체):
    - 전체 실행시간 측정해 runtime.txt에 저장 (속도 평가 대응)
    """
    rows = []
    t0 = _time.time()
    for pid in pids:
        txt = build_summary(int(pid))
        rows.append({"환자번호": int(pid), "summary": txt})
    pd.DataFrame(rows, columns=RESULT_COLS).to_csv(out_csv, index=False, encoding="utf-8-sig")
    Path("runtime.txt").write_text(f"{_time.time()-t0:.2f}", encoding="utf-8")

def write_runtime():
    """노트북 전체 실행시간(T0 기준) → runtime.txt 기록"""
    elapsed = _time.time() - T0
    Path("runtime.txt").write_text(f"{elapsed:.2f}", encoding="utf-8")

# 10) Main (운영진은 아래만 호출: 경로 수정 없이 1회 실행) --------------------
# - 공지대로: ./test 에서 admission.csv 로 전체 환자 처리 → result.csv, runtime.txt 생성
# - 이후 제출 버킷 업로드 시도
def _main():
    # admission의 모든 환자번호 수집 (결측/비정상 제외)
    pids = (
        admission["환자번호"]
        .dropna()
        .astype("int64")
        .tolist()
    )
    if not pids:
        raise RuntimeError("admission.csv에서 환자번호를 찾을 수 없습니다.")
    run_submit_batch(pids, out_csv="result.csv")
    # 전체 실행시간(프로그램 레벨)도 기록
    write_runtime()
    # 제출 버킷 업로드 시도
    if UPLOAD_TO_S3:
        for fn in ("result.csv", "runtime.txt"):
            upload_to_s3_with_retry(fn, attempts=2)
        if DEBUG:
            print(f"[S3] Tried upload -> {_s3_uri('result.csv')} / {_s3_uri('runtime.txt')}")

if __name__ == "__main__":
    _main()
