import re
import json
import ast
import time
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
import requests
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from Config.config import gemini_api


class GeminiMatcher(object):
    def __init__(self, api_key: str = gemini_api, model_name: str = "gemini-2.5-flash",
                 rpm_limit: int = 9, max_retries: int = 5):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("Missing API key. Pass api_key or set environment variable GEMINI_API_KEY.")
        self.model_name = model_name
        self.rpm_limit = max(1, rpm_limit)
        self.max_retries = max(1, max_retries)
        self._ticks = deque()

    # --- Rate limiting (RPM) -------------------------------------------------
    def _respect_rpm_limit(self):
        now = time.time()
        while self._ticks and now - self._ticks[0] >= 60:
            self._ticks.popleft()
        if len(self._ticks) >= self.rpm_limit:
            wait_s = 60 - (now - self._ticks[0]) + 0.05
            if wait_s > 0:
                time.sleep(wait_s)
        self._ticks.append(time.time())

    # --- Retry helpers -------------------------------------------------------
    @staticmethod
    def _retry_delay_from_error(err: ClientError) -> Optional[float]:
        """Parse RetryInfo retryDelay from Google error if present (e.g., '22s')."""
        j = getattr(err, "response_json", None) or {}
        try:
            details = j.get("error", {}).get("details", [])
            for d in details:
                if d.get("@type", "").endswith("RetryInfo"):
                    delay = d.get("retryDelay", "")
                    if delay.endswith("s"):
                        return float(delay[:-1])
                    return float(delay)
        except Exception:
            pass
        return None

    def _is_transient_error(self, err: Exception) -> bool:
        """Heuristic: decide whether error is transient (worth retrying / safe to degrade)."""
        msg = str(err).lower()
        return any(tok in msg for tok in [
            "503", "unavailable", "overloaded", "temporarily", "timeout",
            "deadline", "429", "rate limit", "connection reset", "reset by peer",
            "eof", "bad gateway", "gateway timeout"
        ])

    # --- Low-level API call --------------------------------------------------
    def _post(self, prompt: str) -> str:
        client = genai.Client(api_key=self.api_key)

        attempt = 0
        while True:
            self._respect_rpm_limit()
            try:
                resp = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                        temperature=0.0
                        # 若可用：response_mime_type="application/json"
                    )
                )
                text = getattr(resp, "text", None)
                if not text:
                    try:
                        text = resp.candidates[0].content.parts[0].text
                    except Exception:
                        text = ""
                return text

            except ClientError as e:
                # 429 or 5xx -> retry with backoff; others -> raise
                if e.status_code == 429 or (500 <= e.status_code < 600):
                    attempt += 1
                    if attempt > self.max_retries:
                        raise ConnectionError(f"Gemini API still failing after retries: {e}") from e
                    delay = self._retry_delay_from_error(e)
                    if delay is None:
                        delay = min(2 ** attempt, 30) + 0.1 * attempt  # backoff + mild jitter
                    time.sleep(delay)
                    continue
                raise

    # --- Robust JSON extraction ---------------------------------------------
    def _parse_first_json_like(self, text: str) -> Any:
        if not isinstance(text, str):
            raise ValueError("Model output is not a string")
        s = text.strip()

        # Strip fenced code blocks if present
        m = re.search(r"```(?:json|javascript|python)?\s*(.*?)```", s, flags=re.I | re.S)
        if m:
            s = m.group(1).strip()

        # Find first '{' or '[' and parse balanced segment
        i_obj, i_arr = s.find("{"), s.find("[")
        starts = [i for i in (i_obj, i_arr) if i != -1]
        if not starts:
            raise ValueError("No JSON-like segment found in model output")
        start = min(starts)
        open_ch = s[start]
        close_ch = "}" if open_ch == "{" else "]"

        depth = 0
        end = None
        in_string = False
        escape = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            else:
                if ch == '"':
                    in_string = True
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
        if end is None:
            raise ValueError("Unbalanced JSON brackets in model output")

        segment = s[start:end].strip()

        # Try strict json, then python literal as fallback
        try:
            return json.loads(segment)
        except Exception:
            obj = ast.literal_eval(segment)
            if isinstance(obj, (dict, list)):
                return obj
            raise ValueError("Parsed non-dict/list type.")

    # --- Call + parse with extra retries for transient parse/API errors ------
    def _call_api_json_like(self, prompt: str) -> Any:
        attempt = 0
        while True:
            try:
                content = self._post(prompt)  # _post already retries 429/5xx
                return self._parse_first_json_like(content)
            except Exception as e:
                attempt += 1
                if attempt <= self.max_retries and self._is_transient_error(e):
                    delay = min(2 ** attempt, 30) + 0.05 * attempt
                    time.sleep(delay)
                    continue
                raise

    # --- Public APIs ---------------------------------------------------------
    def test_for_equivalence(self, DICT: Dict[str, str]) -> Dict[str, str]:
        # 高精度版 Prompt，严格保持你原本的结构与行数
        prompt = (
            "Ontology matching.\n"
            "Task: From dict D (key->value), keep pairs that denote the same or near-same concept. Use only D; if unsure, drop.\n"
            "Rules: do not modify any key/value; only keep or delete; case-insensitive; ignore underscores and punctuation; prefer technical semantics over surface overlap; exclude broader/narrower, part/whole, instance/type, device/function, unit/quantity, class/property; enforce one-to-one by keeping the single best match per side.\n"
            "Output ONLY a JSON object (double quotes) of the kept pairs. No extra text. If none, output {}.\n"
            f"D = {json.dumps(DICT, ensure_ascii=True)}"
        )
        try:
            obj = self._call_api_json_like(prompt)
            if not isinstance(obj, dict):
                raise ValueError("Model output is not a JSON object (dict).")
            return obj

        except ClientError as e:
            # transient errors -> graceful degrade
            if e.status_code == 429 or (500 <= e.status_code < 600):
                return {}
            raise

        except (requests.exceptions.RequestException, ConnectionError):
            # network errors -> graceful degrade
            return {}

        except Exception as e:
            # if looks transient -> degrade, else raise parse error
            if self._is_transient_error(e):
                return {}
            raise ValueError(f"Failed to parse response: {e}") from e

    def get_hierarchy_matched_pair(self, hierarchy1_list: List[str],
                                   hierarchy2_list: List[str]) -> List[Tuple[str, str]]:
        """
        句子级匹配：输入两边的 '(child is a subclass of parent)' 句子列表，
        返回含义相同/近似的句子对列表 [(L1_sentence, L2_sentence), ...]
        """
        prompt = (
            "Ontology matching (sentence level).\n"
            "Input: L1 and L2 are lists of facts '(child is a subclass of parent)'.\n"
            "Task: find sentence pairs (L1 vs L2) that mean the same or nearly the same. Use only the given strings; if unsure, skip.\n"
            "Rules: compare whole sentences; consider both child and parent; case-insensitive; "
            "ignore underscores/punctuation/plurals; one L2 per L1; do not change any text.\n"
            "Output ONLY a JSON array of pairs [[\"L1_sentence\",\"L2_sentence\"], ...]. No extra text, no code fences.\n"
            f"L1 = {json.dumps(hierarchy1_list, ensure_ascii=True)}\n"
            f"L2 = {json.dumps(hierarchy2_list, ensure_ascii=True)}"
        )
        try:
            obj = self._call_api_json_like(prompt)
            if not isinstance(obj, list):
                raise ValueError("Model output is not a JSON array.")
            pairs: List[Tuple[str, str]] = []
            for item in obj:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    pairs.append((str(item[0]), str(item[1])))
            return pairs

        except ClientError as e:
            if e.status_code == 429 or (500 <= e.status_code < 600):
                return []
            raise
        except (requests.exceptions.RequestException, ConnectionError):
            return []
        except Exception as e:
            if self._is_transient_error(e):
                return []
            raise ValueError(f"Failed to parse response: {e}") from e

    def get_name_matched_pair(self, L1: List[str], L2: List[str]) -> List[Tuple[str, str]]:
        prompt = (
            "Ontology matching (name-based).\n"
            "Input: L1 and L2 are lists of labels.\n"
            "Task: match by NAME ONLY.\n"
            "Normalize: case-insensitive; treat underscores/hyphens/spaces as equivalent; "
            "ignore punctuation; allow simple singular/plural and minor spacing variants.\n"
            "Rules: one best L2 per L1; use only the given strings; if unsure, skip.\n"
            "Output ONLY a JSON array of pairs [[\"L1_label\",\"L2_label\"], ...]. No extra text.\n"
            f"L1 = {json.dumps(L1, ensure_ascii=True)}\n"
            f"L2 = {json.dumps(L2, ensure_ascii=True)}"
        )
        try:
            obj = self._call_api_json_like(prompt)
            if not isinstance(obj, list):
                raise ValueError("Model output is not a JSON array.")
            pairs: List[Tuple[str, str]] = []
            for item in obj:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    pairs.append((str(item[0]), str(item[1])))
            return pairs

        except ClientError as e:
            if e.status_code == 429 or (500 <= e.status_code < 600):
                return []


if __name__ == '__main__':
    gemini = GeminiMatcher()  # 或 GeminiMatcher(api_key="YOUR_API_KEY")
    dict1 = {
        'Review': 'Review', 'Co-author': 'Contribution_co-author', 'Paper': 'Paper', 'Meta-Reviewer': 'Reviewer',
        'Person': 'Person', 'Reviewer': 'Reviewer', 'Meta-Review': 'Reviewer', 'Chairman': 'Reviewer',
        'Author': 'Regular_author', 'Document': 'Conference_document', 'AssociatedChair': 'Conference_participant',
        'User': 'Committee_member', 'ConferenceChair': 'Conference_volume', 'Administrator': 'Program_committee',
        'ConferenceMember': 'Conference_participant', 'ExternalReviewer': 'Reviewer', 'Conference': 'Conference',
        'AuthorNotReviewer': 'Contribution_co-author', 'PaperAbstract': 'Paper', 'writtenBy': 'reviews',
        'rejectedBy': 'reviews', 'readByMeta-Reviewer': 'reviews', 'hasCo-author': 'reviews',
        'assignedTo': 'issues', 'acceptedBy': 'reviews', 'hasBid': 'has_members', 'detailsEnteredBy': 'reviews',
        'reviewCriteriaEnteredBy': 'reviews', 'assignedByAdministrator': 'invited_by', 'email': 'has_an_email',
        'name': 'has_a_name'
    }
    print(gemini.test_for_equivalence(dict1))
