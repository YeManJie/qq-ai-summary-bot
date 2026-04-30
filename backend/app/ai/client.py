import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


# 自动加载 backend/.env
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class LLMClient:
    def __init__(self):
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.base_url = os.getenv("LLM_BASE_URL", "").strip()
        self.model = os.getenv("LLM_MODEL", "").strip()

        if not self.api_key:
            raise ValueError("缺少环境变量 LLM_API_KEY")
        if not self.base_url:
            raise ValueError("缺少环境变量 LLM_BASE_URL")
        if not self.model:
            raise ValueError("缺少环境变量 LLM_MODEL")

    def _extract_json_text(self, text: str) -> str:
        if not text:
            return ""

        raw = text.strip()

        if (raw.startswith("{") and raw.endswith("}")) or (
            raw.startswith("[") and raw.endswith("]")
        ):
            return raw

        fenced_json_match = re.search(
            r"```json\s*(.*?)\s*```",
            raw,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if fenced_json_match:
            return fenced_json_match.group(1).strip()

        fenced_match = re.search(r"```\s*(.*?)\s*```", raw, flags=re.DOTALL)
        if fenced_match:
            candidate = fenced_match.group(1).strip()
            if candidate:
                return candidate

        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return raw[first_brace:last_brace + 1].strip()

        first_bracket = raw.find("[")
        last_bracket = raw.rfind("]")
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            return raw[first_bracket:last_bracket + 1].strip()

        return raw

    def _try_parse_json(self, text: str):
        cleaned = self._extract_json_text(text)

        try:
            return json.loads(cleaned), cleaned, None
        except Exception as e1:
            first_error = str(e1)

        fallback = cleaned.strip()
        fallback = re.sub(r"^```json\s*", "", fallback, flags=re.IGNORECASE)
        fallback = re.sub(r"^```\s*", "", fallback)
        fallback = re.sub(r"\s*```$", "", fallback)
        fallback = fallback.strip()

        try:
            return json.loads(fallback), fallback, None
        except Exception as e2:
            second_error = str(e2)

        first_brace = fallback.find("{")
        last_brace = fallback.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            obj_candidate = fallback[first_brace:last_brace + 1].strip()
            try:
                return json.loads(obj_candidate), obj_candidate, None
            except Exception as e3:
                third_error = str(e3)
        else:
            third_error = "No valid JSON object braces found"

        parse_error = {
            "first_parse_error": first_error,
            "second_parse_error": second_error,
            "third_parse_error": third_error,
        }
        return None, fallback, parse_error

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        timeout: int = 120,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
        }

        last_error = None

        for attempt in range(2):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                )
                response.raise_for_status()
                data = response.json()

                content = data["choices"][0]["message"]["content"]
                parsed_json, cleaned_text, parse_error = self._try_parse_json(content)

                return {
                    "raw_text": content,
                    "cleaned_text": cleaned_text,
                    "parsed_json": parsed_json,
                    "parse_ok": parsed_json is not None,
                    "parse_error": parse_error,
                    "usage": data.get("usage"),
                    "model": data.get("model", self.model),
                }

            except requests.exceptions.ReadTimeout as e:
                last_error = e
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise

        raise last_error