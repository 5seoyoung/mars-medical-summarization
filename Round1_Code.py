from typing import Any, Dict
import re
import pandas as pd
from processor import DatathonProcessor  # 대회 제공 베이스

# ===== 공통 유틸 =====
def _prepend_meta_if_available(row: Any, text: str) -> str:
    """gender/anchor_age가 있으면 간단 메타를 붙인다(평가상 불이익 없도록 최소화)."""
    if isinstance(row, pd.Series):
        parts = []
        if "gender" in row and pd.notna(row["gender"]):
            parts.append(f"Gender: {row['gender']}")
        if "anchor_age" in row and pd.notna(row["anchor_age"]):
            parts.append(f"Age: {row['anchor_age']}")
        meta = " | ".join(parts)
        return (meta + "\n" if meta else "") + text
    return text


class TaskAProcessor(DatathonProcessor):
    """Task A: Brief Hospital Course"""

    def get_model_name(self) -> str:
        # 허용 모델 정확 표기
        return "meta-llama/Llama-3.1-8B-Instruct"

    def get_prompt_template(self) -> str:
        return (
            "You are a hospitalist. Based on the information below, write a Brief Hospital Course.\n"
            "- Summarize key events chronologically: presentation, major tests/treatments, response, key decisions, complications.\n"
            "- Be concise, clinical tone, no redundancy.\n\n"
            "{user_input}\n\n"
            "Write 5–9 sentences in English."
        )

    async def preprocess_data(self, data: Any) -> Dict[str, str]:
        # 정확한 컬럼명: 'medical record'
        raw = str(data['medical record']).strip()
        return {"user_input": _prepend_meta_if_available(data, raw)}

    async def postprocess_result(self, result: str) -> str:
        return result.strip()


class TaskBProcessor(DatathonProcessor):
    """Task B: Radiology Impression"""

    def get_model_name(self) -> str:
        return "meta-llama/Llama-3.1-8B-Instruct"

    def get_prompt_template(self) -> str:
        return (
            "You are a radiologist. From the following radiology FINDINGS, write the IMPRESSION only.\n"
            "- 1–3 short bullet points with clinically meaningful conclusions.\n"
            "- Be specific (present/absent), avoid hedging.\n\n"
            "{user_input}\n\n"
            "Output ONLY the impression bullets in English (no 'IMPRESSION:' header)."
        )

    async def preprocess_data(self, data: Any) -> Dict[str, str]:
        # 정확한 컬럼명: 'radiology report'
        raw = str(data['radiology report']).strip()
        return {"user_input": _prepend_meta_if_available(data, raw)}

    async def postprocess_result(self, result: str) -> str:
        txt = result.strip()
        # 만약 모델이 "IMPRESSION:"를 붙였다면 제거
        txt = re.sub(r'^\s*IMPRESSION\s*:?\s*', '', txt, flags=re.IGNORECASE)
        return txt.strip()


class TaskCProcessor(DatathonProcessor):
    """Task C: ICD-10 Code Prediction"""

    def get_model_name(self) -> str:
        # 두 모델 모두 허용이나, 여기서는 EXAONE로 다양성 확보
        return "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct-AWQ"

    def get_prompt_template(self) -> str:
        return (
            "You are a medical coder. Based on the discharge course below, predict ICD-10 discharge diagnosis codes.\n"
            "- List 1–2 primary and 0–4 secondary codes total (concise, no overcoding).\n"
            "- Use valid ICD-10 codes only. Output codes separated by commas, no explanations.\n"
            "- Format examples: I10, E119, I479\n\n"
            "{user_input}\n\n"
            "Output: codes only, comma-separated (no titles/descriptions)."
        )

    async def preprocess_data(self, data: Any) -> Dict[str, str]:
        # 정확한 컬럼명: 'hospital_course'  (icd_title 사용 금지)
        raw = str(data['hospital_course']).strip()
        return {"user_input": _prepend_meta_if_available(data, raw)}

    async def postprocess_result(self, result: str) -> str:
        txt = result.strip().upper()
        # 코드 패턴 추출 (문자+숫자 조합, 점 포함 허용 후 점 제거)
        codes = re.findall(r"[A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?", txt)
        # 점 제거하여 I10, E119 형식으로 통일 + 원순서 중복 제거
        norm = []
        seen = set()
        for c in codes:
            c2 = c.replace(".", "")
            if c2 not in seen:
                seen.add(c2)
                norm.append(c2)
        return ", ".join(norm)
