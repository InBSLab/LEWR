import os
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum



class LLMConfirmerConfig:

    

    HIGH_CONFIDENCE_THRESHOLD = 0.55
    LOW_CONFIDENCE_THRESHOLD = 0.3
    

    DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
    DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    DEEPSEEK_MODEL = "deepseek-chat"
    

    TEMPERATURE = 0.3
    MAX_TOKENS = 2500
    

    ENABLE_CACHE = True
    CACHE_SIZE = 500



class ConfidenceLevel(Enum):

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConfirmationResult:
    """确认结果"""
    confirmed: bool
    api: Dict
    confidence: float
    confidence_level: ConfidenceLevel
    llm_called: bool
    reason: str
    alternatives: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'confirmed': self.confirmed,
            'api': self.api,
            'confidence': self.confidence,
            'confidence_level': self.confidence_level.value,
            'llm_called': self.llm_called,
            'reason': self.reason,
            'alternatives': self.alternatives
        }



class DeepSeekClient:

    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv(LLMConfirmerConfig.DEEPSEEK_API_KEY_ENV)
        self._client = None
    
    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=LLMConfirmerConfig.DEEPSEEK_BASE_URL
                )
            except ImportError:
                raise ImportError("请安装openai库: pip install openai")
        return self._client
    
    def generate(self, prompt: str) -> str:

        client = self._get_client()
        response = client.chat.completions.create(
            model=LLMConfirmerConfig.DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=LLMConfirmerConfig.TEMPERATURE,
            max_tokens=LLMConfirmerConfig.MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    
    def is_available(self) -> bool:
        return bool(self.api_key)


class MockLLMClient:

    
    def __init__(self):
        self.call_count = 0
    
    def generate(self, prompt: str) -> str:
        self.call_count += 1
        # 简单规则：默认返回YES
        if 'unrelated' in prompt.lower() or 'wrong' in prompt.lower():
            return "NO"
        return "YES"
    
    def is_available(self) -> bool:
        return True



class LLMConfirmer:

    

    CONFIRM_PROMPT = '''You are an expert in BPMN workflow design and API integration.

WORKFLOW CONTEXT:
User's overall requirement: "{user_request}"

CURRENT TASK:
- Name: {task_name}
- Description: {task_desc}
- Type: {task_type}
- Purpose: This task is part of the above workflow and needs to be automated or supported by an API

CANDIDATE API:
- Name: {api_name}
- Description: {api_desc}
- Core Function: {api_function}

EVALUATION CRITERIA:
1. Functional Match: Does the API's core function align with what the task needs to accomplish?
2. Common Usage: Is this API commonly used for similar tasks in business workflows?
3. Business Fit: Would a business process analyst recommend this API for automating this task?

JUDGMENT GUIDELINES:
- If the API is highly suitable (>80% functional match): Answer YES
- If the API is moderately suitable (50-80% functional match): Answer YES
- If the API is barely suitable or unsuitable (<50% match): Answer NO

Think through each criterion, then provide your final answer.

Final Answer (YES/NO only):'''


    MULTI_CONFIRM_PROMPT = '''You are an expert in BPMN workflow design and API integration.

WORKFLOW CONTEXT:
User's overall requirement: "{user_request}"

CURRENT TASK:
- Name: {task_name}
- Description: {task_desc}
- Type: {task_type}
- Purpose: This task is part of the above workflow and needs to be automated or supported by an API

CANDIDATE APIs (ordered by initial retrieval score — this order is only a hint,
the correct answer is frequently NOT candidate #1):
{candidates_block}

INSTRUCTIONS:
1. Read every candidate before deciding.
2. Pick the single candidate whose core function best accomplishes the CURRENT TASK.
3. Only answer NONE if genuinely none of the candidates is a reasonable functional
   match — do not answer NONE just because the best candidate is imperfect.

Respond in exactly this format, nothing else:
CHOICE: <candidate number, or NONE>
REASON: <one short sentence>'''

    def __init__(
        self,
        api_database: Dict = None,
        use_mock: bool = False,
        high_threshold: float = None,
        low_threshold: float = None,
        api_key: str = None
    ):

        self.api_db = api_database or {}
        self.high_threshold = high_threshold or LLMConfirmerConfig.HIGH_CONFIDENCE_THRESHOLD
        self.low_threshold = low_threshold or LLMConfirmerConfig.LOW_CONFIDENCE_THRESHOLD
        

        if use_mock:
            self.llm = MockLLMClient()
        else:
            self.llm = DeepSeekClient(api_key=api_key)
            if not self.llm.is_available():
                print("[警告] DeepSeek API不可用，使用模拟客户端")
                self.llm = MockLLMClient()
        

        self._cache: Dict[str, bool] = {}

        self._cache_multi: Dict[str, Optional[int]] = {}
        

        self.stats = {
            'total': 0,
            'high_skipped': 0,
            'llm_calls': 0,
            'cache_hits': 0,
            'llm_call_failures': 0  # single-attempt calls (no retry) that raised
                                     # and fell back to a safe default; see
                                     # REPRODUCIBILITY.md §2
        }
    
    def confirm(
        self,
        task_spec: Dict,
        top_candidates: List[Dict],
        user_request: str = ""
    ) -> ConfirmationResult:

        self.stats['total'] += 1
        

        if not top_candidates:
            return ConfirmationResult(
                confirmed=False,
                api={},
                confidence=0.0,
                confidence_level=ConfidenceLevel.LOW,
                llm_called=False,
                reason='No candidates provided'
            )
        
        top1 = top_candidates[0]
        confidence = top1.get('final_score', top1.get('score', 0))
        level = self._get_confidence_level(confidence)
        alternatives = top_candidates[1:3] if len(top_candidates) > 1 else []
        

        if level == ConfidenceLevel.HIGH:
            self.stats['high_skipped'] += 1
            return ConfirmationResult(
                confirmed=True,
                api=top1,
                confidence=confidence,
                confidence_level=level,
                llm_called=False,
                reason='High confidence - LLM skipped',
                alternatives=alternatives
            )
        

        is_suitable = self._llm_confirm(task_spec, top1, user_request)
        
        if is_suitable:
            return ConfirmationResult(
                confirmed=True,
                api=top1,
                confidence=confidence,
                confidence_level=level,
                llm_called=True,
                reason='LLM confirmed',
                alternatives=alternatives
            )
        

        if len(top_candidates) > 1:
            second = top_candidates[1]
            is_suitable_2 = self._llm_confirm(task_spec, second, user_request)
            
            if is_suitable_2:
                return ConfirmationResult(
                    confirmed=True,
                    api=second,
                    confidence=second.get('final_score', second.get('score', 0)),
                    confidence_level=self._get_confidence_level(
                        second.get('final_score', second.get('score', 0))
                    ),
                    llm_called=True,
                    reason='Fallback to second candidate',
                    alternatives=[top1] + top_candidates[2:4]
                )
        

        return ConfirmationResult(
            confirmed=False,
            api=top1,
            confidence=confidence,
            confidence_level=level,
            llm_called=True,
            reason='LLM rejected all candidates',
            alternatives=alternatives
        )
    
    def confirm_multi(
        self,
        task_spec: Dict,
        top_candidates: List[Dict],
        user_request: str = "",
        max_candidates: int = 8
    ) -> ConfirmationResult:

        self.stats['total'] += 1

        if not top_candidates:
            return ConfirmationResult(
                confirmed=False,
                api={},
                confidence=0.0,
                confidence_level=ConfidenceLevel.LOW,
                llm_called=False,
                reason='No candidates provided'
            )

        top1 = top_candidates[0]
        confidence = top1.get('final_score', top1.get('score', 0))
        level = self._get_confidence_level(confidence)

        if level == ConfidenceLevel.HIGH:
            self.stats['high_skipped'] += 1
            return ConfirmationResult(
                confirmed=True,
                api=top1,
                confidence=confidence,
                confidence_level=level,
                llm_called=False,
                reason='High confidence - LLM skipped',
                alternatives=top_candidates[1:3]
            )

        candidates = top_candidates[:max_candidates]

        cache_key = self._cache_key(task_spec, top1) + f"|multi{len(candidates)}"
        if cache_key in self._cache_multi:
            self.stats['cache_hits'] += 1
            cached_idx = self._cache_multi[cache_key]
            return self._build_multi_result(
                cached_idx, top1, confidence, level, candidates,
                top_candidates, llm_called=False,
                reason_prefix='LLM multi-candidate choice (cached)'
            )

        self.stats['llm_calls'] += 1
        choice_idx, reason = self._llm_confirm_multi(task_spec, candidates, user_request)

        if LLMConfirmerConfig.ENABLE_CACHE:
            self._update_cache_multi(cache_key, choice_idx)

        return self._build_multi_result(
            choice_idx, top1, confidence, level, candidates,
            top_candidates, llm_called=True,
            reason_prefix='LLM multi-candidate choice'
        )

    def _build_multi_result(
        self, choice_idx: Optional[int], top1: Dict, confidence: float,
        level: 'ConfidenceLevel', candidates: List[Dict], top_candidates: List[Dict],
        llm_called: bool, reason_prefix: str, reason_detail: str = ''
    ) -> ConfirmationResult:

        if choice_idx is None:
            return ConfirmationResult(
                confirmed=False,
                api=top1,
                confidence=confidence,
                confidence_level=level,
                llm_called=llm_called,
                reason=f'LLM rejected all candidates' + (f': {reason_detail}' if reason_detail else ''),
                alternatives=top_candidates[1:3]
            )

        chosen = candidates[choice_idx - 1]
        chosen_score = chosen.get('final_score', chosen.get('score', confidence))
        return ConfirmationResult(
            confirmed=True,
            api=chosen,
            confidence=chosen_score,
            confidence_level=self._get_confidence_level(chosen_score),
            llm_called=llm_called,
            reason=f'{reason_prefix}' + (f': {reason_detail}' if reason_detail else ''),
            alternatives=[c for c in candidates if c is not chosen][:2]
        )

    def _llm_confirm_multi(
        self, task_spec: Dict, candidates: List[Dict], user_request: str
    ) -> Tuple[Optional[int], str]:

        task_name = task_spec.get('name', task_spec.get('task_name', ''))
        task_desc = task_spec.get('description', 'No description provided')
        task_type = task_spec.get('type', 'serviceTask')

        lines = []
        for i, c in enumerate(candidates, 1):
            api_id = c.get('api_id', c.get('id', ''))
            api_name = c.get('api_name', api_id)
            api_info = self.api_db.get(api_id, self.api_db.get(api_name, {}))
            api_desc = api_info.get('description', c.get('description', 'No description available'))
            api_function = self._extract_api_function(api_desc)
            lines.append(f"{i}. {api_name} — {api_function}")
        candidates_block = '\n'.join(lines)

        prompt = self.MULTI_CONFIRM_PROMPT.format(
            user_request=user_request,
            task_name=task_name,
            task_desc=task_desc,
            task_type=task_type,
            candidates_block=candidates_block
        )

        print(f"      [LLM多候选重排] {task_name} ({task_type}) - {len(candidates)}个候选...")

        try:
            response = self.llm.generate(prompt)
            choice_idx, reason = self._parse_multi_response(response, len(candidates))
            print(f"      [LLM多候选重排] 结果: {choice_idx if choice_idx else 'NONE'}")
            return choice_idx, reason
        except Exception as e:
            print(f"      [LLM多候选重排失败] {e}")
            self.stats['llm_call_failures'] += 1

            return 1, f'LLM call failed, fallback to top1: {e}'

    def _parse_multi_response(self, response: str, num_candidates: int) -> Tuple[Optional[int], str]:

        text = (response or '').strip()
        upper = text.upper()

        reason_match = re.search(r'REASON\s*:\s*(.*)', text, re.IGNORECASE)
        reason = reason_match.group(1).strip() if reason_match else ''

        choice_match = re.search(r'CHOICE\s*:\s*(\d+|NONE)', upper)
        if choice_match:
            val = choice_match.group(1)
            if val == 'NONE':
                return None, reason
            idx = int(val)
            if 1 <= idx <= num_candidates:
                return idx, reason


        for n in re.findall(r'\d+', upper):
            idx = int(n)
            if 1 <= idx <= num_candidates:
                return idx, reason or 'fallback: parsed from unstructured response'

        if 'NONE' in upper:
            return None, reason or 'fallback: unstructured NONE'


        return 1, reason or 'fallback: unparsable response, default to top candidate'

    def _update_cache_multi(self, key: str, value: Optional[int]):

        if len(self._cache_multi) >= LLMConfirmerConfig.CACHE_SIZE:
            first_key = next(iter(self._cache_multi))
            del self._cache_multi[first_key]
        self._cache_multi[key] = value

    def rerank(
        self,
        task_description: str,
        candidates: List[Dict],
        context: str = ""
    ) -> Optional[Dict[str, Any]]:

        if not candidates:
            return None


        top_candidates = []
        for c in candidates:
            top_candidates.append({
                'api_id': c.get('api_id'),
                'api_name': c.get('api_name', c.get('api_id')),
                'description': c.get('description', ''),
                'final_score': c.get('initial_score', c.get('final_score', c.get('score', 0)))
            })


        task_spec = {
            'task_id': None,
            'name': task_description,
            'task_name': task_description,
            'description': task_description,
            'type': 'serviceTask'
        }


        result = self.confirm_multi(
            task_spec=task_spec,
            top_candidates=top_candidates,
            user_request=context
        )

        if not result.confirmed:

            return None

        chosen_api = result.api or {}
        return {
            'api_id': chosen_api.get('api_id'),
            'api_name': chosen_api.get('api_name', chosen_api.get('api_id')),
            'confidence': result.confidence,
            'confirmed': result.confirmed,
            'reason': result.reason
        }

    def batch_confirm(
        self,
        task_specs: List[Dict],
        candidates_per_task: Dict[str, List[Dict]],
        user_request: str
    ) -> Dict[str, ConfirmationResult]:

        results = {}
        for task_spec in task_specs:
            task_id = task_spec.get('task_id')
            candidates = candidates_per_task.get(task_id, [])
            results[task_id] = self.confirm(task_spec, candidates, user_request)
        return results
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:

        if confidence >= self.high_threshold:
            return ConfidenceLevel.HIGH
        elif confidence >= self.low_threshold:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def _llm_confirm(
        self,
        task_spec: Dict,
        api: Dict,
        user_request: str
    ) -> bool:


        cache_key = self._cache_key(task_spec, api)
        if cache_key in self._cache:
            self.stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        self.stats['llm_calls'] += 1
        

        task_name = task_spec.get('name', task_spec.get('task_name', ''))
        task_desc = task_spec.get('description', 'No description provided')
        task_type = task_spec.get('type', 'serviceTask')
        
        api_name = api.get('api_name', api.get('name', ''))
        api_id = api.get('api_id', api.get('id', ''))
        

        api_info = self.api_db.get(api_id, self.api_db.get(api_name, {}))
        api_desc = api_info.get('description', api.get('description', 'No description available'))
        

        api_function = self._extract_api_function(api_desc)
        
        prompt = self.CONFIRM_PROMPT.format(
            task_name=task_name,
            task_desc=task_desc,
            task_type=task_type,
            api_name=api_name,
            api_desc=api_desc,
            api_function=api_function,
            user_request=user_request
        )
        
        print(f"      [LLM确认] {task_name} ({task_type}) → {api_name}...")
        
        try:
            response = self.llm.generate(prompt)
            result = self._parse_response(response)
            print(f"      [LLM确认] 结果: {'YES' if result else 'NO'}")
        except Exception as e:
            print(f"      [LLM确认失败] {e}")
            self.stats['llm_call_failures'] += 1
            result = True
        

        if LLMConfirmerConfig.ENABLE_CACHE:
            self._update_cache(cache_key, result)
        
        return result
    
    def _extract_api_function(self, api_desc: str) -> str:

        if not api_desc or api_desc == 'No description available':
            return "Unknown function"
        

        sentences = api_desc.split('.')
        if sentences:
            core_function = sentences[0].strip()

            if len(core_function) < 20 and len(sentences) > 1:
                core_function = f"{sentences[0].strip()}. {sentences[1].strip()}"
            return core_function
        
        return api_desc
    
    def _parse_response(self, response: str) -> bool:

        response_upper = response.upper().strip()
        

        lines = response_upper.split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line == 'YES' or line.startswith('YES'):
                return True
            if line == 'NO' or line.startswith('NO'):
                return False
        

        if 'YES' in response_upper:
            return True
        if 'NO' in response_upper:
            return False
        

        return True
    
    def _cache_key(self, task_spec: Dict, api: Dict) -> str:

        return f"{task_spec.get('task_id', '')}|{api.get('api_id', api.get('id', ''))}"
    
    def _update_cache(self, key: str, value: bool):

        if len(self._cache) >= LLMConfirmerConfig.CACHE_SIZE:

            first_key = next(iter(self._cache))
            del self._cache[first_key]
        self._cache[key] = value
    
    def get_statistics(self) -> Dict:

        total = self.stats['total']
        return {
            **self.stats,
            'llm_call_rate': self.stats['llm_calls'] / total if total > 0 else 0,
            'skip_rate': self.stats['high_skipped'] / total if total > 0 else 0
        }



def confirm_api_match(
    task_spec: Dict,
    candidates: List[Dict],
    user_request: str,
    api_database: Dict = None,
    use_mock: bool = True
) -> ConfirmationResult:

    confirmer = LLMConfirmer(api_database=api_database, use_mock=use_mock)
    return confirmer.confirm(task_spec, candidates, user_request)



def _test():
    """简单测试"""
    print("=" * 60)
    print("LLM Confirmer (M.2) - 优化版测试")
    print("=" * 60)
    

    api_db = {
        'api_001': {
            'name': 'OpenWeather API',
            'description': 'Real-time weather data and forecasts. Provides current conditions, hourly and daily forecasts.'
        },
        'api_002': {
            'name': 'Google Maps API',
            'description': 'Maps and navigation services. Offers geocoding, routing, and place search.'
        }
    }
    

    confirmer = LLMConfirmer(api_database=api_db, use_mock=True)
    

    print("\n[测试1] 高置信度匹配 (>0.55)")
    task1 = {
        'task_id': 't1', 
        'name': '查询天气', 
        'description': '获取天气预报',
        'type': 'serviceTask'
    }
    candidates1 = [
        {'api_id': 'api_001', 'api_name': 'OpenWeather API', 'final_score': 0.92}
    ]
    result1 = confirmer.confirm(task1, candidates1, "查询北京天气")
    print(f"  确认: {result1.confirmed}")
    print(f"  LLM调用: {result1.llm_called}")
    print(f"  原因: {result1.reason}")
    

    print("\n[测试2] 中置信度匹配 (0.3-0.55)")
    task2 = {
        'task_id': 't2', 
        'name': '路线规划', 
        'description': '导航到目的地',
        'type': 'serviceTask'
    }
    candidates2 = [
        {'api_id': 'api_002', 'api_name': 'Google Maps API', 'final_score': 0.45}
    ]
    result2 = confirmer.confirm(task2, candidates2, "规划去机场的路线")
    print(f"  确认: {result2.confirmed}")
    print(f"  LLM调用: {result2.llm_called}")
    print(f"  原因: {result2.reason}")
    

    print("\n[测试3] 低置信度匹配 (<0.3)")
    task3 = {
        'task_id': 't3', 
        'name': '发送短信', 
        'description': '发送通知',
        'type': 'userTask'
    }
    candidates3 = [
        {'api_id': 'api_001', 'api_name': 'OpenWeather API', 'final_score': 0.25}
    ]
    result3 = confirmer.confirm(task3, candidates3, "发送订单确认")
    print(f"  确认: {result3.confirmed}")
    print(f"  置信度级别: {result3.confidence_level.value}")
    print(f"  LLM调用: {result3.llm_called}")
    

    print("\n[测试4] 批量确认")
    tasks = [task1, task2]
    all_candidates = {
        't1': candidates1,
        't2': candidates2
    }
    batch_results = confirmer.batch_confirm(tasks, all_candidates, "综合请求")
    for tid, res in batch_results.items():
        print(f"  任务{tid}: confirmed={res.confirmed}, llm_called={res.llm_called}")
    

    print("\n[测试5] 统计信息")
    stats = confirmer.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✓ 测试完成！")
    print("=" * 60)
    print("\n优化说明:")
    print("1. HIGH_CONFIDENCE_THRESHOLD: 0.8 → 0.55 (减少不必要的LLM调用)")
    print("2. TEMPERATURE: 0.1 → 0.3 (增加LLM灵活性)")
    print("3. 改进Prompt: 提供完整上下文和评估标准")
    print("4. 传递任务类型: 帮助LLM更好判断")
    print("5. API功能提取: 突出核心功能")


if __name__ == "__main__":
    _test()
