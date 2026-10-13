# -*- coding: utf-8 -*-


import os
import re
import json
import time
import copy
import hashlib
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Stage 1：增强版 BPMN 生成器（单次 LLM 调用 + 结构校验 + 兜底模板）
from enhanced_bpmn_generator import EnhancedBPMNGenerator


try:
    from simcse_encoder import SimCSEEncoder, create_encoder_from_meta
    HAS_ENCODER = True
except ImportError:
    HAS_ENCODER = False
    print("⚠️ SimCSEEncoder未加载，API推荐将使用mock模式")


try:
    from task_enhancer import TaskEnhancer
    HAS_ENHANCER = True
except ImportError:
    HAS_ENHANCER = False
    print("⚠️ TaskEnhancer未加载，将跳过任务语义增强（d' = d）")



TASK_ELEMENT_TYPES = {
    'userTask', 'serviceTask', 'sendTask', 'receiveTask',
    'manualTask', 'businessRuleTask', 'scriptTask', 'task',
}


TASK_TYPE_PREFIX = {
    'serviceTask': 'Automated API task:',
    'userTask': 'Manual user task:',
}


DEFAULT_TOP_K = 30
KEYWORD_BONUS_PER_HIT = 0.10
KEYWORD_BONUS_CAP = 0.30
DOMAIN_BONUS = 0.20
OFF_DOMAIN_PENALTY = 0.30


HIGH_SCORE_THRESHOLD = 0.55
LOW_SCORE_THRESHOLD = 0.30
MEDIUM_MAX_CHECK = 1
LOW_MAX_CHECK = 2


CALIB_KEYWORD_BONUS = 0.10
CALIB_FLOOR_TRIGGER = 0.45
CALIB_FLOOR_VALUE = 0.50


_RULE_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are',
    'of', 'it', 'its', 'his', 'her', 'their', 'with', 'from', 'by', 'as', 'be', 'will',
    'can', 'this', 'that', 'then', 'into', 'out', 'up', 'if', 'all', 'any', 'one',
    'new', 'case', 'get', 'make',
}


DEFAULT_EXCLUDE_PATTERNS = [
    r'game|gaming|esports',
    r'video|movie|entertainment|music|song|audio',
    r'social|facebook|twitter',
]



def _tokenize(text: str) -> List[str]:

    return re.findall(r'[a-z0-9]+', (text or '').lower())


def _task_keywords(text: str) -> set:

    return {w for w in _tokenize(text) if len(w) > 2 and w not in _RULE_STOPWORDS}


def _names_from_table(content) -> List[str]:

    if isinstance(content, list):
        return [str(x.get('name', '')) if isinstance(x, dict) else str(x) for x in content]
    if isinstance(content, dict):
        keys = list(content.keys())
        if keys and all(str(k).isdigit() for k in keys):
            ordered = sorted(keys, key=lambda k: int(k))
            return [str(content[k].get('name', '')) if isinstance(content[k], dict) else str(content[k])
                    for k in ordered]
        return [str(k) for k in keys]
    return []


def _desc_fields(item) -> Tuple[str, str, List[str]]:

    if isinstance(item, dict):
        desc = ''
        for k in ('description', 'desc', 'text', 'summary', 'tokens'):
            if item.get(k):
                v = item[k]
                desc = ' '.join(str(w) for w in v) if isinstance(v, list) else str(v)
                break
        cat = item.get('category', '') or item.get('categories', '') or ''
        if isinstance(cat, list):
            cat = ' '.join(str(c) for c in cat)
        tags = item.get('tags', []) or []
        if not isinstance(tags, list):
            tags = [str(tags)]
        return desc, str(cat), tags
    if isinstance(item, list):
        return ' '.join(str(w) for w in item), '', []
    return (str(item) if item else ''), '', []


def _format_gateway_context(gc: Optional[Dict[str, Any]]) -> str:

    if not gc or not gc.get('gateway_id'):
        return "none (the task is on the main sequence flow, not inside any gateway branch)"
    text = (f"inside {gc.get('gateway_type') or 'gateway'} '{gc['gateway_id']}', "
            f"branch index {gc.get('branch_index')}")
    cond = (gc.get('branch_condition') or '').strip()
    if cond:
        text += f", branch condition: \"{cond}\""
    text += f", nesting depth {gc.get('depth', 0)}"
    return text



class AdaptiveConfirmer:


    SYSTEM_PROMPT = (
        "You are an expert in business process management and web API selection. "
        "You verify whether a candidate API can realise a given workflow task. "
        "Always answer with exactly one word: YES or NO."
    )

    def __init__(
        self,
        api_database: Dict[str, Any],
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
        max_attempts: int = 3,
        debug: bool = True,
    ):
        self.api_database = api_database
        self.api_key = api_key or ''
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip('/')
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.debug = debug

        self._cache: Dict[str, bool] = {}
        self.stats = {'llm_calls': 0, 'cache_hits': 0, 'accepted': 0, 'rejected': 0, 'failed': 0}

    def is_available(self) -> bool:
        return bool(self.api_key)

    # ---------------------------- 提示词构造 ---------------------------- #

    @staticmethod
    def _core_function_sentence(description: str, name: str) -> str:

        text = ' '.join((description or '').split())
        if not text:
            return name
        first = re.split(r'(?<=[.!?])\s+', text, maxsplit=1)[0]
        words = first.split()
        return ' '.join(words[:40]) if len(words) > 40 else first

    def build_prompt(self, task: Dict[str, Any], api_id: str, workflow_description: str) -> str:

        info = self.api_database.get(api_id, {})
        api_name = info.get('name', api_id)
        api_desc = ' '.join((info.get('description') or api_name).split())
        if len(api_desc) > 1000:
            api_desc = api_desc[:1000] + ' ...'
        core = self._core_function_sentence(info.get('description', ''), api_name)

        task_name = task.get('original_name') or task.get('name', '')
        enhanced_desc = task.get('description', '') or task_name
        task_type = task.get('type', 'task')
        gateway_ctx = _format_gateway_context(task.get('gateway_context'))

        return f"""Decide whether the candidate API is a suitable implementation for the workflow task.

[1. Overall workflow description]
{(workflow_description or '').strip()[:2000]}

[2. Current task]
- Name: {task_name}
- Enhanced description: {enhanced_desc}
- BPMN task type: {task_type}
- Gateway context: {gateway_ctx}

[3. Candidate API]
- Name: {api_name}
- Full description: {api_desc}
- Core function: {core}

[4. Evaluation criteria]
(a) Functional match: the API's function matches what the task needs to accomplish.
(b) Common usage in business workflows: the API is commonly used for this kind of step.
(c) Overall business process fit: the API fits the surrounding process, including the gateway branch condition if any.

Answer with exactly one word: YES or NO."""

    # ---------------------------- LLM 调用 ---------------------------- #

    def _chat(self, prompt: str) -> Optional[str]:

        if not self.is_available():
            return None
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 2500,
            "stream": False,
        }
        for attempt in range(self.max_attempts):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()['choices'][0]['message']['content']

                if resp.status_code in (429, 500, 502, 503, 504):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if self.debug:
                    print(f"    ⚠️ LLM请求失败，HTTP {resp.status_code}")
                return None
            except Exception as e:
                if self.debug:
                    print(f"    ⚠️ LLM请求异常（第{attempt + 1}次）: {type(e).__name__}")
                time.sleep(1.0 * (attempt + 1))
        return None

    @staticmethod
    def _parse_yes_no(text: Optional[str]) -> Optional[bool]:

        if not text:
            return None
        t = text.strip().upper()
        if re.match(r'^\W*YES\b', t):
            return True
        if re.match(r'^\W*NO\b', t):
            return False
        has_yes = re.search(r'\bYES\b', t) is not None
        has_no = re.search(r'\bNO\b', t) is not None
        if has_yes and not has_no:
            return True
        if has_no and not has_yes:
            return False
        return None

    # ---------------------------- 对外接口 ---------------------------- #

    def confirm(self, task: Dict[str, Any], api_id: str, workflow_description: str) -> Optional[bool]:

        prompt = self.build_prompt(task, api_id, workflow_description)


        key = hashlib.sha1(prompt.encode('utf-8')).hexdigest()
        if key in self._cache:
            self.stats['cache_hits'] += 1
            return self._cache[key]

        self.stats['llm_calls'] += 1
        verdict = self._parse_yes_no(self._chat(prompt))
        if verdict is None:
            self.stats['failed'] += 1
            return None

        self._cache[key] = verdict
        self.stats['accepted' if verdict else 'rejected'] += 1
        return verdict



class APIRecommender:


    def __init__(
        self,
        api_database: Dict[str, Any],
        api_embeddings: Optional[np.ndarray] = None,
        api_ids: Optional[List[str]] = None,
        encoder: Optional[Any] = None,
        confirmer: Optional[AdaptiveConfirmer] = None,
        enhancer: Optional[Any] = None,
        top_k: int = DEFAULT_TOP_K,
        high_threshold: float = HIGH_SCORE_THRESHOLD,
        low_threshold: float = LOW_SCORE_THRESHOLD,

        encode_enhanced_text: bool = True,

        rules_use_enhanced_text: bool = False,
        use_exclude_penalty: bool = True,
        exclude_patterns: Optional[List[str]] = None,
        debug: bool = True,
    ):
        self.api_database = api_database if api_database else {}
        self.api_embeddings = api_embeddings
        self.api_ids = list(api_ids) if api_ids else []
        self.encoder = encoder
        self.confirmer = confirmer
        self.enhancer = enhancer
        self.top_k = top_k
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.encode_enhanced_text = encode_enhanced_text
        self.rules_use_enhanced_text = rules_use_enhanced_text
        self.use_exclude_penalty = use_exclude_penalty
        self.exclude_patterns = list(exclude_patterns) if exclude_patterns is not None \
            else list(DEFAULT_EXCLUDE_PATTERNS)
        self.fine_k = 10
        self.debug = debug
        self._domain_table_cache: Optional[Dict[str, List[str]]] = None

        self.stats = {
            'total_recommendations': 0,
            'enhanced_tasks': 0,
            'direct_accepted': 0,
            'llm_confirmed_tasks': 0,
            'llm_rejected_tasks': 0,
            'llm_skipped': 0,
            'mock_fallbacks': 0,
        }

    def get_stats(self) -> Dict[str, Any]:

        merged = dict(self.stats)
        if self.confirmer is not None:
            merged['confirmer'] = dict(self.confirmer.stats)
        return merged


    def recommend_for_tasks(
        self,
        tasks: List[Dict[str, Any]],
        user_request: str = ""
    ) -> Dict[str, Dict[str, Any]]:

        api_bindings: Dict[str, Dict[str, Any]] = {}

        if not tasks:
            if self.debug:
                print("⚠️ 没有可推荐API的任务")
            return api_bindings

        if self.debug:
            print(f"\n{'=' * 70}")
            print(f"Recommending APIs for {len(tasks)} tasks")
            print(f"{'=' * 70}")

        for idx, task in enumerate(tasks, 1):
            task_id = task.get('task_id', task.get('id', f'task_{idx}'))
            if self.debug:
                print(f"\n[{idx}/{len(tasks)}] Task: {task.get('name', '')} (ID: {task_id})")

            enhanced_task = self._enhance_task(task)                       # Stage 3
            recommendation = self._recommend_single_task(enhanced_task, user_request)  # Stage 4-5

            api_bindings[task_id] = recommendation
            self.stats['total_recommendations'] += 1

            if self.debug:
                print(f"  ✓ Recommended: {recommendation.get('api_name', 'unknown')}")
                print(f"    - Confidence (comp*): {recommendation.get('confidence', 0):.3f}")
                print(f"    - Method: {recommendation.get('recommendation_method', 'unknown')}")

        if self.debug:
            unique_apis = len({b.get('api_name', '') for b in api_bindings.values()})
            print(f"\n{'=' * 70}")
            print(f"  ✓ Generated {len(api_bindings)} API recommendations")
            print(f"  ✓ Unique APIs: {unique_apis}/{len(api_bindings)}")

        return api_bindings


    def _enhance_task(self, task: Dict) -> Dict:

        task_name = task.get('name', '')
        task_desc = task.get('description') or task_name

        enhanced = dict(task)
        enhanced['original_name'] = task_name
        enhanced['original_description'] = task_desc
        enhanced['description'] = task_desc

        if self.enhancer is not None:
            try:
                result = self.enhancer.enhance(dict(enhanced))
                if isinstance(result, dict):
                    enhanced = result
                self.stats['enhanced_tasks'] += 1
            except Exception as e:
                if self.debug:
                    print(f"  ⚠️ 任务语义增强失败，使用原始描述: {e}")
                enhanced = dict(task)
                enhanced['description'] = task_desc


        enhanced['original_name'] = task_name
        enhanced['original_description'] = task_desc
        for key in ('task_id', 'id', 'type', 'gateway_context'):
            if key not in enhanced and key in task:
                enhanced[key] = task[key]
        if not (enhanced.get('description') or '').strip():
            enhanced['description'] = task_desc
        return enhanced


    def _mock_binding(self, task_id: str, task_name: str, reason: str) -> Dict[str, Any]:

        self.stats['mock_fallbacks'] += 1
        return {
            'api_id': f'mock-api-{task_id}',
            'api_name': f'mock-api-{task_id}',
            'confidence': 0.0,
            'task_name': task_name,
            'recommendation_method': 'mock',
            'error': reason,
        }

    def _build_retrieval_query(self, task: Dict) -> str:

        if self.encode_enhanced_text:
            body = (task.get('description') or '').strip()
        else:
            name = (task.get('original_name') or task.get('name') or '').strip()
            desc = (task.get('original_description') or '').strip()
            body = f"{name}. {desc}" if desc and desc.lower() != name.lower() else name
        prefix = TASK_TYPE_PREFIX.get(task.get('type', ''), '')
        return f"{prefix} {body}".strip()

    def _recommend_single_task(self, task: Dict, user_request: str) -> Dict:

        task_id = task.get('task_id', task.get('id', 'unknown'))
        task_name = task.get('original_name') or task.get('name', '')

        if self.api_embeddings is None or self.encoder is None:
            return self._mock_binding(task_id, task_name, 'encoder or embeddings unavailable')


        query = self._build_retrieval_query(task)
        try:
            task_vec = self.encoder.encode(query)
            similarities = self.encoder.compute_similarity(
                task_vec, self.api_embeddings,
                top_k=self.top_k, target_ids=self.api_ids
            )
        except Exception as e:
            if self.debug:
                print(f"  ⚠️ 向量检索失败 [{task_id}]: {e}")
            return self._mock_binding(task_id, task_name, f'retrieval failed: {e}')

        if not similarities:
            return self._mock_binding(task_id, task_name, 'no candidates retrieved')


        clean_name = (task.get('original_name') or task_name or '').strip()
        clean_desc = (task.get('original_description') or '').strip()
        clean_text = f"{clean_name} {clean_desc}" if clean_desc and clean_desc.lower() != clean_name.lower() \
            else clean_name
        enhanced_text = f"{clean_name} {task.get('description', '')}"
        rule_text = enhanced_text if self.rules_use_enhanced_text else clean_text

        ranked = self._rule_based_rerank(rule_text, similarities)


        best_api, best_score, method, verdict = self._tiered_confirmation(task, ranked, user_request)


        api_name = self.api_database.get(best_api, {}).get('name', best_api)
        comp = self._calibrate_confidence(best_score, rule_text, api_name)

        return {
            'api_id': best_api,
            'api_name': api_name,
            'confidence': float(comp),
            'task_name': task_name,
            'recommendation_method': method,
            'llm_verdict': verdict,
            'top_candidates': [
                {
                    'api_id': api_id,
                    'api_name': self.api_database.get(api_id, {}).get('name', api_id),
                    'score': float(score)
                }
                for api_id, score in ranked[:self.fine_k]
            ]
        }


    def _tiered_confirmation(
        self,
        task: Dict,
        ranked: List[Tuple[str, float]],
        user_request: str
    ) -> Tuple[str, float, str, str]:

        top_id, top_score = ranked[0]

        if top_score >= self.high_threshold:
            self.stats['direct_accepted'] += 1
            return top_id, top_score, 'vector_search_enhanced', 'not_needed'

        if self.confirmer is None or not self.confirmer.is_available():
            self.stats['llm_skipped'] += 1
            return top_id, top_score, 'vector_search_enhanced', 'skipped'

        max_check = MEDIUM_MAX_CHECK if top_score >= self.low_threshold else LOW_MAX_CHECK
        tier = 'medium' if top_score >= self.low_threshold else 'low'
        if self.debug:
            print(f"    ↳ {tier}-score tier (s'={top_score:.3f}) → LLM confirmation")

        for rank in range(min(max_check, len(ranked))):
            cand_id, cand_score = ranked[rank]
            verdict = self.confirmer.confirm(task, cand_id, user_request)

            if verdict is None:
                self.stats['llm_skipped'] += 1
                return top_id, top_score, 'vector_search_enhanced', 'skipped'

            cand_name = self.api_database.get(cand_id, {}).get('name', cand_id)
            if self.debug:
                print(f"    ↳ LLM {'YES' if verdict else 'NO'} for rank-{rank + 1}: {cand_name}")
            if verdict:
                self.stats['llm_confirmed_tasks'] += 1
                return cand_id, cand_score, 'llm_rerank', 'confirmed'

        self.stats['llm_rejected_tasks'] += 1
        return top_id, top_score, 'llm_rerank', 'rejected'


    def _get_domain_table(self) -> Dict[str, List[str]]:

        if self._domain_table_cache is not None:
            return self._domain_table_cache

        table: Dict[str, List[str]] = {}
        if self.enhancer is not None and getattr(self.enhancer, 'DOMAIN_KEYWORDS', None):
            top_words = getattr(self.enhancer, 'CATEGORY_TOP_WORDS', {}) or {}
            for category in self.enhancer.DOMAIN_KEYWORDS.keys():
                tag_tokens = {t for t in re.split(r'[-_\s]+', str(category).lower()) if t}
                kws = tag_tokens | {str(w).lower() for w in (top_words.get(category) or [])}
                kws = {k for k in kws if len(k) > 2 and k not in _RULE_STOPWORDS}
                if kws:
                    table[category] = sorted(kws)
        self._domain_table_cache = table
        return table

    def _triggered_domain_keywords(self, task_text: str) -> List[List[str]]:

        table = self._get_domain_table()
        if not table:
            return []
        tokens = _tokenize(task_text)
        triggered = []
        for kws in table.values():
            if any(tok.startswith(kw) for kw in kws for tok in tokens):
                triggered.append(kws)
        return triggered

    @staticmethod
    def _name_has_domain_keyword(api_name_lower: str, triggered: List[List[str]]) -> bool:

        return any(kw in api_name_lower for kws in triggered for kw in kws)

    def _rule_based_rerank(self, task_text: str, candidates: list) -> list:

        task_keywords = _task_keywords(task_text)
        triggered = self._triggered_domain_keywords(task_text)
        exclude_patterns = self.exclude_patterns if self.use_exclude_penalty else []

        scored = []
        for api_id, base_score in candidates:
            info = self.api_database.get(api_id, {})
            api_name = info.get('name', api_id)
            api_desc = info.get('description', api_name)
            api_name_lower = str(api_name).lower()
            api_words = set(_tokenize(f"{api_name} {api_desc}"))


            hits = len(task_keywords & api_words)
            bonus = min(KEYWORD_BONUS_PER_HIT * hits, KEYWORD_BONUS_CAP)


            if triggered and self._name_has_domain_keyword(api_name_lower, triggered):
                bonus += DOMAIN_BONUS


            penalty = 0.0
            for pattern in exclude_patterns:
                if re.search(pattern, api_name_lower):
                    penalty = OFF_DOMAIN_PENALTY
                    break

            scored.append((api_id, base_score + bonus - penalty))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(api_id, min(max(score, 0.0), 1.0)) for api_id, score in scored]


    def _calibrate_confidence(self, raw_score: float, task_text: str, api_name: str) -> float:

        name_tokens = set(_tokenize(api_name))
        delta_kw = CALIB_KEYWORD_BONUS if (_task_keywords(task_text) & name_tokens) else 0.0
        calibrated = raw_score + delta_kw

        delta_floor = 0.0
        if calibrated < CALIB_FLOOR_TRIGGER:
            triggered = self._triggered_domain_keywords(task_text)
            if triggered and self._name_has_domain_keyword(str(api_name).lower(), triggered):
                delta_floor = CALIB_FLOOR_VALUE - calibrated

        return min(1.0, calibrated + delta_floor)


    def add_api(self, name: str, description: str = '', category: str = '',
                tags: Optional[List[str]] = None) -> bool:

        if self.encoder is None or self.api_embeddings is None:
            raise RuntimeError("编码器或向量矩阵不可用，无法新增API")
        if name in self.api_database:
            return False

        doc = f"{name}. {description}".strip() if description else name
        vec = np.asarray(self.encoder.encode(doc), dtype=self.api_embeddings.dtype).reshape(1, -1)
        if vec.shape[1] != self.api_embeddings.shape[1]:
            raise ValueError(f"向量维度不一致: {vec.shape[1]} vs {self.api_embeddings.shape[1]}")

        self.api_embeddings = np.vstack([self.api_embeddings, vec])
        self.api_ids.append(name)
        self.api_database[name] = {
            'name': name, 'description': description, 'category': category,
            'tags': tags or [], 'row': len(self.api_ids) - 1,
        }
        return True



class CompleteWorkflowGenerator:


    def __init__(
        self,
        api_key: str,
        api_database_path: Optional[Path] = None,
        api_description_path: Optional[Path] = None,
        api_embeddings_path: Optional[Path] = None,
        api_ids_path: Optional[Path] = None,
        enable_api_recommendation: bool = True,
        top_k: int = DEFAULT_TOP_K,
        high_threshold: float = HIGH_SCORE_THRESHOLD,
        low_threshold: float = LOW_SCORE_THRESHOLD,
        encode_enhanced_text: bool = True,
        debug: bool = True,
        **legacy_kwargs
    ):

        if legacy_kwargs and debug:
            print(f"⚠️ 以下参数在论文版本中已移除，将被忽略: {sorted(legacy_kwargs)}")

        self.api_key = api_key
        self.debug = debug


        self.bpmn_generator = EnhancedBPMNGenerator(api_key=api_key, debug=debug)

        self.enable_api_recommendation = False
        self.api_recommender: Optional[APIRecommender] = None

        if not (enable_api_recommendation and (api_database_path or api_embeddings_path)):
            return

        if self.debug:
            print("\n" + "=" * 70)
            print("Initializing API Recommendation System")
            print("=" * 70)


        api_ids: List[str] = []
        if api_ids_path and api_ids_path.exists():
            try:
                with open(api_ids_path, 'r', encoding='utf-8') as f:
                    api_ids = _names_from_table(json.load(f))
                if self.debug:
                    print(f"  ✓ api_ids: {len(api_ids)} IDs loaded")
            except Exception as e:
                print(f"  ✗ api_ids: Failed to load ({e})")

        api_database = self._load_api_database(api_database_path, api_description_path, api_ids)


        api_embeddings = None
        v2_meta, center_vec = None, None
        emb_dir = api_embeddings_path.parent if api_embeddings_path else None
        if emb_dir is not None:
            v2_npy = emb_dir / "api_features_v2.npy"
            v2_meta_f = emb_dir / "api_features_v2_meta.json"
            v2_mean_f = emb_dir / "api_features_v2_mean.npy"
            if v2_npy.exists() and v2_meta_f.exists():
                try:
                    api_embeddings = np.load(v2_npy)
                    with open(v2_meta_f, 'r', encoding='utf-8') as f:
                        v2_meta = json.load(f)
                    if v2_meta.get('centered') and v2_mean_f.exists():
                        center_vec = np.load(v2_mean_f)
                    print(f"  ✓ 使用重建的 v2 向量表: {v2_npy.name} {api_embeddings.shape} "
                          f"(backend={v2_meta.get('backend')}, model={v2_meta.get('model_name')}, "
                          f"pooling={v2_meta.get('pooling')}, centered={bool(v2_meta.get('centered'))})")
                except Exception as e:
                    print(f"  ✗ v2 向量表加载失败，回退旧 npy ({e})")
                    api_embeddings, v2_meta, center_vec = None, None, None

        if api_embeddings is None and api_embeddings_path and api_embeddings_path.exists():
            try:
                api_embeddings = np.load(api_embeddings_path)
                print(f"  ✓ 使用向量表 {api_embeddings_path.name} {api_embeddings.shape}")
                print("  ⚠️ 若检索结果接近随机，说明该向量表与当前编码器不匹配，"
                      "请用 rebuild_api_embeddings.py 重建 api_features_v2.npy")
            except Exception as e:
                print(f"  ✗ api_embeddings: Failed to load ({e})")

        if api_embeddings is not None and api_ids and len(api_ids) != len(api_embeddings):
            print(f"  ❌ ERROR: ids 数量({len(api_ids)}) 与向量行数({len(api_embeddings)}) 不一致，"
                  f"API 推荐结果不可信！")

        if not (api_database and api_embeddings is not None):
            if self.debug:
                print("\n⚠️ API recommendation disabled (no API database)")
            return

        self.enable_api_recommendation = True


        encoder = None
        if HAS_ENCODER:
            try:
                encoder = create_encoder_from_meta(v2_meta, center_vec=center_vec) \
                    if v2_meta is not None else SimCSEEncoder()
                if self.debug:
                    print("  ✓ Encoder initialized")
            except Exception as e:
                if self.debug:
                    print(f"  ⚠️ Failed to initialize encoder: {e}")


        confirmer = AdaptiveConfirmer(api_database=api_database, api_key=api_key, debug=debug)
        if self.debug:
            print(f"  ✓ LLM Confirmer initialized (model={confirmer.model})")


        enhancer = None
        if HAS_ENHANCER:
            try:
                enhancer = TaskEnhancer()
                if self.debug:
                    print("  ✓ Task Enhancer initialized")
            except Exception as e:
                if self.debug:
                    print(f"  ⚠️ Failed to initialize enhancer: {e}")

        self.api_recommender = APIRecommender(
            api_database=api_database,
            api_embeddings=api_embeddings,
            api_ids=api_ids,
            encoder=encoder,
            confirmer=confirmer,
            enhancer=enhancer,
            top_k=top_k,
            high_threshold=high_threshold,
            low_threshold=low_threshold,
            encode_enhanced_text=encode_enhanced_text,
            debug=debug
        )

        if self.debug:
            print(f"\n✓ API Recommender initialized with {len(api_database)} APIs (K={top_k})")


    def _load_api_database(
        self,
        api_name_path: Optional[Path],
        api_description_path: Optional[Path],
        api_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:

        print("\n[API Files Status]")

        names_data: List[str] = []
        if api_name_path and api_name_path.exists():
            try:
                with open(api_name_path, 'r', encoding='utf-8') as f:
                    names_data = _names_from_table(json.load(f))
                print(f"  ✓ api_name.json: {len(names_data)} APIs loaded")
            except Exception as e:
                print(f"  ✗ api_name.json: Failed to load ({e})")
        else:
            print("  ✗ api_name.json: Not found")

        ids = [str(x) for x in api_ids] if api_ids else list(names_data)
        if not ids:
            print("  ✗ 没有可用的 API id 列表，API数据库为空")
            return {}

        if api_ids and names_data:
            if len(names_data) == len(ids):
                agree = sum(1 for a, b in zip(names_data, ids) if a == b) / len(ids)
                print(f"  ✓ data/api_name.json 与 embeddings/api_name.json 逐行一致率: {agree:.1%}")
                if agree < 0.99:
                    print("  ⚠️ ERROR: 两份名字表的行顺序不一致，行号对齐不可信，请核对！")
            else:
                print(f"  ⚠️ ERROR: data/api_name.json({len(names_data)}) 与 "
                      f"embeddings ids({len(ids)}) 长度不同，行号对齐不可信，请核对！")

        api_database: Dict[str, Any] = {}
        dup = 0
        for i, api_id in enumerate(ids):
            if api_id in api_database:
                dup += 1
                continue
            api_database[api_id] = {'name': api_id, 'description': '', 'category': '',
                                    'tags': [], 'row': i}
        if dup:
            print(f"  ⚠️ {dup} 个重复的API名字（只保留首次出现的行的描述）")

        matched = 0
        if api_description_path and api_description_path.exists():
            try:
                with open(api_description_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    raw = {str(i): item for i, item in enumerate(raw)}

                keys = list(raw.keys())
                digit_frac = (sum(1 for k in keys if str(k).isdigit()) / len(keys)) if keys else 0.0
                by_row = digit_frac > 0.9
                print(f"  ✓ api_description.json: {len(raw)} entries, "
                      f"key 模式 = {'行号' if by_row else '名字'}")

                for key, item in raw.items():
                    if by_row:
                        idx = int(key)
                        if not (0 <= idx < len(ids)):
                            continue
                        api_id = ids[idx]
                        if api_database.get(api_id, {}).get('row') != idx:
                            continue
                    else:
                        api_id = str(key)
                        if api_id not in api_database:
                            continue
                    desc, cat, tags = _desc_fields(item)
                    if desc.strip():
                        info = api_database[api_id]
                        info['description'], info['category'], info['tags'] = desc, cat, tags
                        matched += 1

                rate = matched / len(api_database)
                print(f"  ✓ 描述对齐成功: {matched}/{len(api_database)} ({rate:.1%})")
                if rate < 0.9:
                    print("  ❌ ERROR: 描述对齐率低于90%！规则重排和LLM确认将只看到API名字。"
                          "请检查 api_description.json 的 key 与 embeddings/api_name.json 的行号/名字是否对应。")
            except Exception as e:
                print(f"  ✗ api_description.json: Failed to load ({e})")
        else:
            print("  ✗ api_description.json: Not found")


        for api_id, info in api_database.items():
            if not info['description'].strip():
                readable = re.sub(r'[-_]+', ' ', str(info['name'])).strip()
                if readable.lower() != str(info['name']).lower():
                    info['description'] = readable

        return api_database


    def generate(
        self,
        process_description: str,
        max_retries: int = 3,
        validate: bool = True
    ) -> Dict[str, Any]:

        if self.debug:
            print("\n" + "=" * 70)
            print("Processing User Request")
            print("=" * 70)
            print(f"Input: {process_description[:200]}...")


        bpmn_result = None
        try:
            bpmn_result = self.bpmn_generator.generate(
                process_description=process_description,
                max_retries=max_retries,
                validate=validate
            )
        except Exception as e:
            print(f"  ⚠️ BPMN生成异常（{type(e).__name__}: {e}），改用顺序模板兜底")


        if bpmn_result is None:
            if self.debug:
                print("  ⚠️ BPMN生成无有效结果，使用顺序模板兜底")
            bpmn_result = self.bpmn_generator._generate_fallback(process_description)


        api_bindings: Dict[str, Any] = {}
        if self.enable_api_recommendation and self.api_recommender:
            tasks = self._extract_tasks_from_bpmn(bpmn_result)    # Stage 2
            if self.debug:
                print(f"\n✓ Extracted {len(tasks)} tasks from workflow")
            if tasks:
                api_bindings = self.api_recommender.recommend_for_tasks(
                    tasks, process_description
                )


        result = {
            "bpmn_workflow": bpmn_result.get("bpmn_workflow", {}),
            "evaluation_format": bpmn_result.get("evaluation_format", {}),
            "enhanced_info": bpmn_result.get("enhanced_info", {}),
            "validation": bpmn_result.get("validation", {"is_valid": True}),
            "api_bindings": api_bindings,
            "api_recommendation_stats": self.api_recommender.get_stats() if self.api_recommender else {},
            "complete_output": self._merge_api_bindings(
                bpmn_result.get("evaluation_format", {}),
                api_bindings
            ),
            "metadata": {
                "generator": "LEWR CompleteWorkflowGenerator (paper-aligned v5.0)",
                "timestamp": bpmn_result.get("metadata", {}).get("timestamp", ""),
                "input_length": len(process_description),
                "has_api_recommendation": len(api_bindings) > 0
            }
        }

        if self.debug:
            print("\n" + "=" * 70)
            print("Generation Complete")
            print("=" * 70)
            print("\n[BPMN Statistics]")
            print(f"  Tasks: {result['enhanced_info'].get('task_count', 0)}")
            print(f"  Gateways: {result['enhanced_info'].get('gateway_count', 0)}")
            print("\n[API Recommendations]")
            print(f"  Total bindings: {len(api_bindings)}")
            print(f"  Unique APIs: {len({b.get('api_name', '') for b in api_bindings.values()})}")

        return result


    @staticmethod
    def _iter_branches(gateway: Dict[str, Any]):

        branches = gateway.get('branches') or []
        gw_conditions = gateway.get('conditions') if isinstance(gateway.get('conditions'), list) else []
        for idx, branch in enumerate(branches):
            condition, path = '', []
            if isinstance(branch, dict):
                condition = branch.get('condition') or branch.get('label') or branch.get('name') or ''
                for key in ('path', 'tasks', 'elements', 'process', 'steps', 'body'):
                    if isinstance(branch.get(key), list):
                        path = branch[key]
                        break
            elif isinstance(branch, list):
                path = branch
            else:
                continue
            if not condition and idx < len(gw_conditions):
                cond_item = gw_conditions[idx]
                condition = cond_item.get('condition', '') if isinstance(cond_item, dict) else str(cond_item)
            yield idx, str(condition or ''), path

    def _dfs_collect(
        self,
        elements: List[Any],
        out: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
        depth: int
    ):

        for element in elements or []:
            if not isinstance(element, dict):
                continue
            etype = element.get('type', '')

            if etype in TASK_ELEMENT_TYPES:
                task_id = element.get('id') or f"task_{len(out) + 1}"
                label = (element.get('label') or element.get('name') or f"Task {len(out) + 1}").strip()
                description = (element.get('description') or label).strip()
                out.append({
                    'task_id': task_id,
                    'id': task_id,
                    'name': label,
                    'description': description,
                    'type': etype,

                    'gateway_context': {
                        'gateway_id': context['gateway_id'] if context else None,
                        'gateway_type': context['gateway_type'] if context else None,
                        'branch_index': context['branch_index'] if context else None,
                        'branch_condition': context['branch_condition'] if context else '',
                        'depth': depth,
                    },
                })
            elif element.get('branches') is not None:   # 网关：递归进入每条分支
                gateway_id = element.get('id') or f"gateway_{depth}_{len(out)}"
                for idx, condition, path in self._iter_branches(element):
                    self._dfs_collect(
                        path, out,
                        {
                            'gateway_id': gateway_id,
                            'gateway_type': etype,
                            'branch_index': idx,
                            'branch_condition': condition,
                        },
                        depth + 1
                    )

    def _extract_tasks_from_bpmn(self, bpmn_result: Dict) -> List[Dict]:

        if bpmn_result is None:
            return []

        process = (bpmn_result.get("bpmn_workflow") or {}).get("process") or []
        tasks: List[Dict[str, Any]] = []
        self._dfs_collect(process, tasks, None, 0)
        if tasks:
            return tasks


        if self.debug:
            print("  ⚠️ 层次化BPMN中未遍历到任务，退回使用evaluation_format的任务列表")
        for task in (bpmn_result.get("evaluation_format") or {}).get("tasks", []):
            name = task.get('name', '')
            tasks.append({
                'task_id': task.get('task_id', ''),
                'id': task.get('task_id', ''),
                'name': name,
                'description': name,
                'type': task.get('type', 'task'),
                'gateway_context': None,
            })
        return tasks



    def _merge_api_bindings(
        self,
        evaluation_format: Dict,
        api_bindings: Dict
    ) -> Dict:

        merged = copy.deepcopy(evaluation_format) if evaluation_format else {}
        if "tasks" not in merged or not api_bindings:
            return merged


        by_name = {}
        for binding in api_bindings.values():
            key = (binding.get('task_name') or '').strip().lower()
            if key and key not in by_name:
                by_name[key] = binding

        for task in merged["tasks"]:
            task_id = task.get('task_id', '')
            binding = api_bindings.get(task_id)
            if binding is None:
                binding = by_name.get((task.get('name') or '').strip().lower())
            if binding is not None:
                task['api_recommendation'] = binding

        return merged
