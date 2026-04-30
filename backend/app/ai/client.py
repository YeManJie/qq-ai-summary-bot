import json
import re
import time
import requests

from app.config import settings


class LLMClient:
    def __init__(self):
        self.api_key = settings.llm_api_key
        self.base_url = settings.llm_base_url
        self.model = settings.llm_model

    def _extract_json_text(self, text: str) -> str:
        """
        尽量从模型输出中提取可用于 json.loads 的 JSON 文本。
        依次处理：
        1. 纯 JSON
        2. ```json ... ``` 代码块
        3. ``` ... ``` 代码块
        4. 从整段文本中截取最外层 {...} 或 [...]
        """
        if not text:
            return ""

        raw = text.strip()

        # 情况1：直接就是 JSON
        if (raw.startswith("{") and raw.endswith("}")) or (raw.startswith("[") and raw.endswith("]")):
            return raw

        # 情况2：```json ... ```
        fenced_json_match = re.search(r"```json\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if fenced_json_match:
            return fenced_json_match.group(1).strip()

        # 情况3：``` ... ```
        fenced_match = re.search(r"```\s*(.*?)\s*```", raw, flags=re.DOTALL)
        if fenced_match:
            candidate = fenced_match.group(1).strip()
            if candidate:
                return candidate

        # 情况4：从文本中提取第一个 JSON 对象 {...}
        first_brace = raw.find("{")
        last_brace = raw.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            return raw[first_brace:last_brace + 1].strip()

        # 情况5：从文本中提取第一个 JSON 数组 [...]
        first_bracket = raw.find("[")
        last_bracket = raw.rfind("]")
        if first_bracket != -1 and last_bracket != -1 and last_bracket > first_bracket:
            return raw[first_bracket:last_bracket + 1].strip()

        return raw

    def _try_parse_json(self, text: str):
        """
        多轮尝试解析 JSON。
        返回:
        - parsed_json
        - cleaned_text
        - parse_error
        """
        cleaned = self._extract_json_text(text)

        # 第一次直接解析
        try:
            return json.loads(cleaned), cleaned, None
        except Exception as e1:
            first_error = str(e1)

        # 第二次：去掉可能残留的 ```json / ```
        fallback = cleaned.strip()
        fallback = re.sub(r"^```json\s*", "", fallback, flags=re.IGNORECASE)
        fallback = re.sub(r"^```\s*", "", fallback)
        fallback = re.sub(r"\s*```$", "", fallback)
        fallback = fallback.strip()

        try:
            return json.loads(fallback), fallback, None
        except Exception as e2:
            second_error = str(e2)

        # 第三次：只截取最外层大括号
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

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
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

        # 最多尝试 2 次，请求超时/偶发波动时重试一次
        for attempt in range(2):
            try:
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=120,
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