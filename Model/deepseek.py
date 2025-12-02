from openai import OpenAI
from Config.config import deepseek_api
import requests
import json
import re
import ast
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any


class DeepSeekMatcher(object):
    def __init__(self, api_key: str = deepseek_api, model_name: str = "deepseek-chat"):
        """
        Initialize the DeepSeek matcher with API credentials

        Args:
            api_key: Your DeepSeek API key (expects a variable deepseek_api defined elsewhere)
            model_name: Name of the model to use (default: "deepseek-chat")
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    # ---------------- Core HTTP + Parsing ----------------
    def _post(self, prompt: str) -> str:
        # Clean Python memory to avoid caching effects
        import gc
        gc.collect()

        # Always start a fresh, context-free message list
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }

        # Call DeepSeek API
        resp = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=120
        )

        resp.raise_for_status()
        data = resp.json()

        # Extract content safely
        try:
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            raise ValueError(f"Unexpected DeepSeek response format: {data}") from e

        if not isinstance(content, str):
            raise ValueError(f"Model output is not string: {content}")

        return content

    def _call_api_plain(self, prompt: str) -> str:
        """
        直接拿模型输出的纯文本，不做任何 JSON 尝试解析。
        """
        content = self._post(prompt)
        if not isinstance(content, str):
            raise ValueError("Model output is not a string")
        return content

    def _parse_first_json_like(self, text: str) -> Any:
        """
        从任意模型输出中提取第一个 JSON 片段（对象或数组），
        先尝试 json.loads，失败则退回 ast.literal_eval。
        支持 ```json ... ``` 代码块、前后缀说明文本等。
        """
        if not isinstance(text, str):
            raise ValueError("Model output is not a string")
        s = text.strip()

        # 先剥离代码围栏 ```json ... ``` / ``` ... ```
        m = re.search(r"```(?:json|python)?\s*(.*?)```", s, flags=re.I | re.S)
        if m:
            s = m.group(1).strip()

        # 找到第一个起始符号 { 或 [
        i_obj = s.find("{")
        i_arr = s.find("[")
        starts = [i for i in (i_obj, i_arr) if i != -1]
        if not starts:
            raise ValueError("No JSON-like segment found in model output")
        start = min(starts)
        open_ch = s[start]
        close_ch = "}" if open_ch == "{" else "]"

        # 括号配对，截取完整片段
        depth = 0
        end = None
        for i in range(start, len(s)):
            ch = s[i]
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end is None:
            raise ValueError("Unbalanced JSON brackets in model output")

        segment = s[start:end]

        # 优先严格 JSON（双引号），失败再退回 Python 字面量（支持单引号）
        try:
            return json.loads(segment)
        except json.JSONDecodeError:
            return ast.literal_eval(segment)

    def _call_api_json_like(self, prompt: str) -> Any:
        content = self._post(prompt)
        return self._parse_first_json_like(content)

    # ---------------- Public APIs ----------------


    def pairwise_equivalence_list(self, DICT: Dict[str, str]) -> List[Tuple[str, str]]:
        """
        输入:
            DICT = {source: target, ...}

        行为:
            对每个 (source, target) 单独问一次模型
            prompt 只有一句话
            如果模型回答 YES，我们就视为同一概念并保留

        输出:
            一个列表: [ (source, target), ... ]
        """

        results: List[Tuple[str, str]] = []

        for src, tgt in DICT.items():
            prompt = (
                f'In the energy domain, do "{src}" and "{tgt}" refer to the same semantic concept? '
                f'Answer ONLY "YES" or "NO".'
            )
            try:
                raw_answer = self._call_api_plain(prompt)
            except requests.exceptions.RequestException as e:
                raise ConnectionError(f"API request failed on pair ({src}, {tgt}): {e}")
            except Exception as e:
                raise ValueError(f"Failed to get model answer for pair ({src}, {tgt}): {e}")

            # 清洗回答，比如 "YES", "YES.", "Yes\n", "YES they are the same"
            ans = raw_answer.strip().upper()
            ans = ans.replace(".", "").replace("!", "").replace("\"", "").strip()

            if ans.startswith("YES"):
                results.append((src, tgt))

        return results
    def pairwise_equivalence_dict(self, DICT: Dict[str, str]) -> Dict[str, str]:
        """
        跑逐对匹配, 但最后返回 dict {source: target} 以兼容下游导出逻辑.

        规则:
        - 我们先用 pairwise_equivalence_list 得到 [("src","tgt"), ...]
        - 再转成字典
        - 如果出现两个 source 指向不同 target 或多个 source 指向同一个 target
          我们简单地采取第一次出现的那个, 后面的忽略
        """

        pairs = self.pairwise_equivalence_list(DICT)

        final: Dict[str, str] = {}
        used_targets = set()

        for src, tgt in pairs:
            if src in final:
                # 这个 source 已经有了, 跳过后续
                continue
            if tgt in used_targets:
                # 这个 target 已经被别的 source 占了, 跳过保持一对一
                continue

            final[src] = tgt
            used_targets.add(tgt)

        return final


    def get_hierarchy_matched_pair(self, hierarchy1_list: List[str], hierarchy2_list: List[str]) -> List[
        Tuple[str, str]]:
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
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"API request failed: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}")

    def get_name_matched_pair(self, L1: List[str], L2: List[str]) -> str:
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
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"API request failed: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}")
    def test_for_equivalence(self, DICT: Dict[str, str]) -> Dict[str, str]:
        """
        输入 {key: value}，让模型按“同义/近义”仅保留匹配项；输出仍为字典。
        召回友好：不确定时倾向保留（keep）。
        """
        D_JSON = json.dumps(DICT, ensure_ascii=False)
        prompt = (
            f"""Ontology Matching — Final Filter (High Precision)

        You are given a dict D of source→target candidates:
        D = {D_JSON}

        Task:
        Keep ONLY pairs that are the SAME or NEAR-IDENTICAL concept.

        Rules:
        1) Case-insensitive; ignore underscores, hyphens, punctuation, plural/singular.
        2) Keep only true synonyms or standard aliases/acronyms.
        3) Drop pairs showing broader/narrower, part-of/has-part, instance/type, function↔device, unit↔quantity, class↔property, or overlapping sets.
        4) Prefer technical meaning over token overlap.
        5) One-to-one: for any source (or target), keep the single best match and drop the rest.
        6) If unsure, DROP (favor precision).

        Output:
        Return ONLY a JSON object mapping kept sources to targets (double quotes). No extra text. If none, return {{}}.
        """
        )

        try:
            obj = self._call_api_json_like(prompt)
            if not isinstance(obj, dict):
                raise ValueError("Model output is not a JSON object (dict).")
            return obj
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"API request failed: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}")


if __name__ == '__main__':
    dp = DeepSeekMatcher()
    # list1 = ['Simulation Tool', 'Co-Simulation Framework Compatibility', 'Device under Test']
    # list2 = ['Hardware', 'Simulation Tool', 'General Parameters', 'Co-Simulation', 'Simulation Adapter']
    # matches = dp.get_matched_pairs(list1, list2, "name")
    dict1 = {
        "Review": "Review",
        "Co-author": "Contribution_co-author",
        "Paper": "paper",
        "Meta-Reviewer": "Reviewer",
        "Person": "Person"
    }
    matches = dp.test_for_equivalence(dict1)
    print(matches)
