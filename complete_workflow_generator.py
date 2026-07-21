
import os
import sys
import json
import re
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# 导入原有的增强BPMN生成器
from enhanced_bpmn_generator import EnhancedBPMNGenerator


try:
    from simcse_encoder import SimCSEEncoder
    HAS_ENCODER = True
except ImportError:
    HAS_ENCODER = False
    print("⚠️ SimCSEEncoder未加载，API推荐将使用模拟模式")

try:
    from llm_confirmer import LLMConfirmer
    HAS_CONFIRMER = True
except ImportError:
    HAS_CONFIRMER = False
    print("⚠️ LLMConfirmer未加载，将跳过LLM重排序")

try:
    from task_enhancer import TaskEnhancer
    HAS_ENHANCER = True
except ImportError:
    HAS_ENHANCER = False
    print("⚠️ TaskEnhancer未加载，将跳过任务增强")




def _estimate_task_range(description: str) -> Tuple[int, int]:

    desc_lower = description.lower()

    branch_markers = [
        'if ', 'otherwise', 'in case', 'depending on', 'either', 'or else',
        'unless', 'when ', 'whenever'
    ]
    sequence_markers = [
        'then', 'after', 'afterwards', 'meanwhile', 'in the meantime',
        'once', 'subsequently', 'next,'
    ]

    branch_count = sum(desc_lower.count(m) for m in branch_markers)
    sequence_count = sum(desc_lower.count(m) for m in sequence_markers)


    sentence_count = max(1, len(re.findall(r'[.!?](?:\s|$)', description)))


    base = sentence_count
    expansion = branch_count * 2 + sequence_count

    min_tasks = max(4, int(base * 0.8))
    max_tasks = max(min_tasks + 4, int((base + expansion) * 1.3))


    max_tasks = min(max_tasks, 30)

    return min_tasks, max_tasks



class APIRecommender:


    def __init__(
        self,
        api_database: Dict[str, Any],
        api_embeddings: Optional[np.ndarray] = None,
        api_ids: Optional[List[str]] = None,
        encoder: Optional[Any] = None,
        confirmer: Optional[Any] = None,
        enhancer: Optional[Any] = None,
        top_k: int = 30,

        rerank_threshold: float = 0.48,

        use_semantic_bridge: bool = False,
        bridge_confidence_threshold: float = 0.40,
        debug: bool = True
    ):
        self.api_database = api_database if api_database else {}
        self.api_embeddings = api_embeddings
        self.api_ids = api_ids if api_ids else []
        self.encoder = encoder
        self.confirmer = confirmer
        self.enhancer = enhancer
        self.top_k = top_k
        self.rerank_threshold = rerank_threshold
        self.use_semantic_bridge = use_semantic_bridge
        self.bridge_confidence_threshold = bridge_confidence_threshold
        self.fine_k = 9
        self.debug = debug

        self.stats = {
            'total_recommendations': 0,
            'enhanced_tasks': 0,
            'llm_reranked': 0,
            'direct_recommendations': 0,
            'semantic_bridged': 0
        }

    def recommend_for_tasks(
        self,
        tasks: List[Dict[str, Any]],
        user_request: str = ""
    ) -> Dict[str, Dict[str, Any]]:

        api_bindings = {}

        if not tasks:
            if self.debug:
                print("⚠️ No tasks to recommend APIs for")
            return api_bindings

        if self.debug:
            print(f"\n{'='*70}")
            print(f"Recommending APIs for {len(tasks)} tasks")
            print(f"{'='*70}")


        used_api_names = {}  # api_name -> count

        for idx, task in enumerate(tasks, 1):
            task_id = task.get('task_id', task.get('id', f'task_{idx}'))
            task_name = task.get('name', '')

            if self.debug:
                print(f"\n[{idx}/{len(tasks)}] Task: {task_name} (ID: {task_id})")

            enhanced_task = self._enhance_task(task)

            recommendation = self._recommend_single_task(
                enhanced_task,
                user_request,
                used_api_names=used_api_names
            )

            if recommendation:
                api_bindings[task_id] = recommendation
                self.stats['total_recommendations'] += 1

                # v3.6: 记录已使用的API
                api_name = recommendation.get('api_name', '')
                if api_name:
                    used_api_names[api_name] = used_api_names.get(api_name, 0) + 1

                if self.debug:
                    api_name = recommendation.get('api_name', 'unknown')
                    confidence = recommendation.get('confidence', 0)
                    method = recommendation.get('recommendation_method', 'unknown')
                    print(f"  ✓ Recommended: {api_name}")
                    print(f"    - Confidence: {confidence:.3f}")
                    print(f"    - Method: {method}")

        if self.debug:
            print(f"\n{'='*70}")
            print(f"  ✓ Generated {len(api_bindings)} API recommendations")
            unique_apis = len(set(b.get('api_name', '') for b in api_bindings.values()))
            print(f"  ✓ Unique APIs: {unique_apis}/{len(api_bindings)}")

        return api_bindings

    def _enhance_task(self, task: Dict) -> Dict:

        enhanced = task.copy()

        task_name = task.get('name', '')
        task_desc = task.get('description', task_name)


        enhanced['original_description'] = task_desc


        words = re.findall(r'\b\w+\b', task_name.lower())
        keywords = [w for w in words if len(w) > 4]
        synonym_map = {
            'send': ['deliver', 'transmit', 'dispatch'],
            'receive': ['accept', 'get', 'obtain'],
            'check': ['verify', 'validate', 'confirm'],
            'create': ['generate', 'make', 'build'],
            'delete': ['remove', 'erase', 'clear'],
            'update': ['modify', 'change', 'edit'],
        }
        expanded_keywords = keywords.copy()
        for keyword in keywords:
            if keyword in synonym_map:
                expanded_keywords.extend(synonym_map[keyword][:2])

        enhanced['keywords'] = expanded_keywords


        clean_original_description = task_desc

        if self.enhancer is not None:
            try:
                enhanced = self.enhancer.enhance(enhanced)
                self.stats['enhanced_tasks'] += 1
            except Exception as e:
                if self.debug:
                    print(f"  ⚠️ Task enhancement failed: {e}")

                enhanced['description'] = (
                    f"{task_desc} {' '.join(expanded_keywords)}" if expanded_keywords else task_desc
                )
        else:

            enhanced['description'] = (
                f"{task_desc} {' '.join(expanded_keywords)}" if expanded_keywords else task_desc
            )

        enhanced['original_description'] = clean_original_description

        return enhanced

    def _recommend_single_task(
        self,
        task: Dict,
        user_request: str,
        used_api_names: Dict[str, int] = None
    ) -> Dict:

        task_id = task.get('task_id', task.get('id', 'unknown'))
        task_name = task.get('name', '')
        task_desc = task.get('description', task_name)

        if self.api_embeddings is None or self.encoder is None:
            return {
                'api_name': f'mock-api-{task_id}',
                'api_id': task_id,
                'confidence': 0.8,
                'task_name': task_name,
                'recommendation_method': 'mock'
            }

        task_text = f"{task_name} {task_desc}"

        retrieval_query = self._build_retrieval_query(task)
        try:
            task_vec = self.encoder.encode(retrieval_query)
        except Exception as e:
            if self.debug:
                print(f"  ⚠️ Encoding failed for [{task_id}]: {e}")
            return {
                'api_name': f'error-{task_id}',
                'api_id': task_id,
                'confidence': 0.0,
                'task_name': task_name,
                'error': str(e)
            }

        try:
            similarities = self.encoder.compute_similarity(
                task_vec,
                self.api_embeddings,
                top_k=self.top_k,
                target_ids=self.api_ids
            )
        except Exception as e:
            if self.debug:
                print(f"  ⚠️ Similarity computation failed for [{task_id}]: {e}")
            return {
                'api_name': f'error-{task_id}',
                'api_id': task_id,
                'confidence': 0.0,
                'task_name': task_name,
                'error': str(e)
            }

        if not similarities:
            return {
                'api_name': f'no-match-{task_id}',
                'api_id': task_id,
                'confidence': 0.0,
                'task_name': task_name,
                'recommendation_method': 'no_match'
            }

        reranked_candidates = self._rule_based_rerank(
            task_text, similarities, used_api_names=used_api_names
        )

        best_api, best_score = reranked_candidates[0]
        top_candidates = reranked_candidates[:self.fine_k]

        recommendation_method = 'vector_search_enhanced'


        if (self.use_semantic_bridge
                and best_score < self.bridge_confidence_threshold
                and self.confirmer is not None):
            bridged_text = self._semantic_bridge(task_text)
            if bridged_text:
                try:
                    bridged_vec = self.encoder.encode(bridged_text)
                    bridged_similarities = self.encoder.compute_similarity(
                        bridged_vec, self.api_embeddings,
                        top_k=self.top_k, target_ids=self.api_ids
                    )
                    if bridged_similarities:
                        bridged_reranked = self._rule_based_rerank(
                            task_text, bridged_similarities, used_api_names=used_api_names
                        )
                        bridged_api, bridged_score = bridged_reranked[0]
                        if bridged_score > best_score:
                            if self.debug:
                                print(f"    ↳ Semantic bridge improved score: "
                                      f"{best_score:.3f} → {bridged_score:.3f} "
                                      f"(bridged: \"{bridged_text[:60]}...\")")
                            best_api, best_score = bridged_api, bridged_score
                            top_candidates = bridged_reranked[:self.fine_k]
                            recommendation_method = 'semantic_bridge'
                            self.stats['semantic_bridged'] += 1
                except Exception as e:
                    if self.debug:
                        print(f"    ⚠️ Semantic bridge retrieval failed: {e}")

        if best_score < self.rerank_threshold and self.confirmer is not None:
            if self.debug:
                print(f"    ↳ Triggering LLM rerank (score={best_score:.3f} < {self.rerank_threshold})")

            try:
                candidate_infos = []
                for api_id, score in top_candidates:
                    api_info = self.api_database.get(api_id, {})
                    api_name = api_info.get('name', api_id)
                    api_desc = api_info.get('description', api_name)

                    candidate_infos.append({
                        'api_id': api_id,
                        'api_name': api_name,
                        'description': api_desc,
                        'initial_score': score
                    })

                llm_result = self.confirmer.rerank(
                    task_description=task_text,
                    candidates=candidate_infos,
                    context=user_request
                )

                if llm_result and 'api_id' in llm_result:
                    best_api = llm_result['api_id']
                    best_score = llm_result.get('confidence', best_score)
                    recommendation_method = 'llm_rerank'
                    self.stats['llm_reranked'] += 1

                    if self.debug:
                        print(f"    ✓ LLM selected: {llm_result.get('api_name', best_api)}")

            except Exception as e:
                if self.debug:
                    print(f"    ⚠️ LLM rerank failed: {e}")

        api_info = self.api_database.get(best_api, {})
        api_name = api_info.get('name', best_api)

        calibrated_confidence = self._calibrate_confidence(best_score, task_text, api_name)

        return {
            'api_id': best_api,
            'api_name': api_name,
            'confidence': float(calibrated_confidence),
            'task_name': task_name,
            'recommendation_method': recommendation_method,
            'top_candidates': [
                {
                    'api_id': api_id,
                    'api_name': self.api_database.get(api_id, {}).get('name', api_id),
                    'score': float(score)
                }
                for api_id, score in top_candidates
            ]
        }

    def _semantic_bridge(self, task_text: str) -> Optional[str]:

        confirmer = self.confirmer
        client = getattr(confirmer, 'llm', None) if confirmer is not None else None
        if client is None or not getattr(client, 'is_available', lambda: False)():
            return None
        # MockLLMClient只返回YES/NO，不是真正的文本生成，桥接到这种客户端
        # 上没有意义，直接跳过比硬塞"YES"当查询文本去检索更安全。
        if type(client).__name__ == 'MockLLMClient':
            return None

        prompt = f"""Convert this workflow task into a web application / API functionality description.

Task: "{task_text}"

Requirements:
1. Describe what functionality is needed, in plain application/API terminology
2. Do NOT mention any specific company or API names (e.g. fedex, stripe, google)
3. Write 15-25 words
4. Output only the description, no quotes, no explanation

Example:
Task: "take parcel insurance"
Output: application provides insurance coverage for packages calculates rates based on value handles claims processing delivery protection

Now convert:
Task: "{task_text}"
Output:"""

        try:
            bridged = client.generate(prompt)
            bridged = (bridged or '').strip().strip('"').strip()
            return bridged if bridged else None
        except Exception as e:
            if self.debug:
                print(f"    ⚠️ Semantic bridge failed: {e}")
            return None

    def _build_retrieval_query(self, task: Dict) -> str:

        task_name = (task.get('name') or '').strip()

        original_desc = (task.get('original_description') or '').strip()

        if original_desc and original_desc.lower() != task_name.lower():
            return f"{task_name}. {original_desc}"
        return task_name

    def _get_auto_domain_pairs(self) -> Dict[str, Dict[str, Any]]:

        cached = getattr(self, '_auto_domain_pairs_cache', None)
        if cached is not None:
            return cached

        if self.enhancer is None or not getattr(self.enhancer, 'DOMAIN_KEYWORDS', None):
            self._auto_domain_pairs_cache = {}
            return self._auto_domain_pairs_cache

        top_words_table = getattr(self.enhancer, 'CATEGORY_TOP_WORDS', {}) or {}
        pairs: Dict[str, Dict[str, Any]] = {}
        for category in self.enhancer.DOMAIN_KEYWORDS.keys():

            tag_tokens = {t for t in re.split(r'[-_\s]+', category) if t}
            keyword_set = tag_tokens | set(top_words_table.get(category) or [])
            if not keyword_set:
                continue
            keyword_list = sorted(keyword_set)
            pairs[category] = {
                'task_keywords': keyword_list,
                'api_keywords': keyword_list,
                'category_tag': category,
            }
        self._auto_domain_pairs_cache = pairs
        return pairs

    def _rule_based_rerank(self, task_text: str, candidates: list,
                           used_api_names: Dict[str, int] = None) -> list:

        task_lower = task_text.lower()
        task_keywords = set(re.findall(r'\b\w+\b', task_lower))
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'is', 'are'}
        task_keywords = task_keywords - stopwords

        domain_pairs = self._get_auto_domain_pairs()

        exclude_patterns = [
            r'game|gaming|esports',
            r'video|movie|entertainment',
            r'music|song|audio',
            r'social|facebook|twitter',
        ]

        weighted_candidates = []

        for api_id, base_score in candidates:
            api_info = self.api_database.get(api_id, {})
            api_name = api_info.get('name', api_id)
            api_desc = api_info.get('description', api_name)

            bonus = 0.0

            api_text = f"{api_name} {api_desc}".lower()
            api_words = set(re.findall(r'\b\w+\b', api_text))

            keyword_overlap = len(task_keywords & api_words)
            if keyword_overlap > 0:
                bonus += 0.10 * min(keyword_overlap, 3)

            api_name_lower = api_name.lower()
            for keyword in task_keywords:
                if keyword in api_name_lower:
                    bonus += 0.10


            api_category_field = str(api_info.get('category', '')).lower()
            for domain, keywords in domain_pairs.items():
                task_match = any(kw in task_lower for kw in keywords['task_keywords'])
                if not task_match:
                    continue

                text_match = any(kw in api_name_lower or kw in api_text for kw in keywords['api_keywords'])
                if not text_match:
                    continue

                category_tag = keywords.get('category_tag', domain)
                category_field_match = bool(category_tag) and category_tag in api_category_field

                bonus += 0.20 + (0.05 if category_field_match else 0.0)
                break

            for pattern in exclude_patterns:
                if re.search(pattern, api_name_lower):
                    bonus -= 0.30
                    break

            patterns = {
                'notification': ['notification', 'email', 'sms', 'alert'],
                'payment': ['payment', 'transaction', 'billing', 'invoice'],
                'storage': ['storage', 'database', 'file', 'document'],
                'authentication': ['auth', 'login', 'user', 'account'],
            }

            for pattern_key, pattern_apis in patterns.items():
                if pattern_key in task_lower:
                    if any(pa in api_name_lower for pa in pattern_apis):
                        bonus += 0.15


            if used_api_names and api_name in used_api_names:
                use_count = used_api_names[api_name]
                diversity_penalty = min(0.08 * use_count, 0.25)
                bonus -= diversity_penalty


            raw_final_score = base_score + bonus
            weighted_candidates.append((api_id, raw_final_score))

        weighted_candidates.sort(key=lambda x: x[1], reverse=True)

        weighted_candidates = [(api_id, min(max(score, 0.0), 1.0)) for api_id, score in weighted_candidates]
        return weighted_candidates

    def _calibrate_confidence(self, raw_score: float, task_text: str, api_name: str) -> float:
        """置信度校准"""
        calibrated = raw_score

        task_words = set(task_text.lower().split())
        api_words = set(api_name.lower().replace('-', ' ').split())
        if task_words & api_words:
            calibrated += 0.10

        if calibrated < 0.45:

            if self.enhancer is not None and getattr(self.enhancer, 'DOMAIN_KEYWORDS', None):
                domain_pairs = self._get_auto_domain_pairs()
                domain_keywords = sorted({
                    kw for info in domain_pairs.values() for kw in info.get('api_keywords', [])
                })
            else:
                domain_keywords = []
            task_lower = task_text.lower()
            api_lower = api_name.lower()

            has_relevance = (task_words & api_words) or any(
                kw in api_lower for kw in domain_keywords if kw in task_lower
            )

            if has_relevance:
                calibrated = 0.50

        return min(1.0, calibrated)



class CompleteWorkflowGenerator:


    def __init__(
        self,
        api_key: str,
        api_database_path: Optional[Path] = None,
        api_description_path: Optional[Path] = None,
        api_embeddings_path: Optional[Path] = None,
        api_ids_path: Optional[Path] = None,
        enable_api_recommendation: bool = True,
        top_k: int = 30,
        rerank_threshold: float = 0.48,
        use_semantic_bridge: bool = False,
        debug: bool = True
    ):
        self.api_key = api_key
        self.debug = debug


        self.bpmn_generator = EnhancedBPMNGenerator(api_key=api_key, debug=debug)


        self.enable_api_recommendation = False
        self.api_recommender = None

        if enable_api_recommendation and (api_database_path or api_embeddings_path):
            if self.debug:
                print("\n" + "="*70)
                print("Initializing API Recommendation System")
                print("="*70)

            api_database = self._load_api_database(api_database_path, api_description_path)

            api_embeddings = None
            if api_embeddings_path and api_embeddings_path.exists():
                try:
                    api_embeddings = np.load(api_embeddings_path)
                    if self.debug:
                        print(f"  ✓ api_embeddings: {api_embeddings.shape}")
                except Exception as e:
                    print(f"  ✗ api_embeddings: Failed to load ({e})")

            api_ids = []
            if api_ids_path and api_ids_path.exists():
                try:
                    with open(api_ids_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                        if isinstance(content, dict):
                            api_ids = list(content.keys())
                        elif isinstance(content, list):
                            api_ids = content
                    if self.debug:
                        print(f"  ✓ api_ids: {len(api_ids)} IDs loaded")
                except Exception as e:
                    print(f"  ✗ api_ids: Failed to load ({e})")

            if api_database and api_embeddings is not None:
                self.enable_api_recommendation = True

                encoder = None
                confirmer = None
                enhancer = None

                if HAS_ENCODER:
                    try:
                        encoder = SimCSEEncoder()
                        if self.debug:
                            print("  ✓ SimCSE Encoder initialized")
                    except Exception as e:
                        if self.debug:
                            print(f"  ⚠️ Failed to initialize encoder: {e}")

                if HAS_CONFIRMER:
                    try:
                        confirmer = LLMConfirmer(api_key=api_key)
                        if self.debug:
                            print("  ✓ LLM Confirmer initialized")
                    except Exception as e:
                        if self.debug:
                            print(f"  ⚠️ Failed to initialize confirmer: {e}")

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
                    rerank_threshold=rerank_threshold,
                    use_semantic_bridge=use_semantic_bridge,
                    debug=debug
                )

                if self.debug:
                    print(f"\n✓ API Recommender initialized with {len(api_database)} APIs")
            else:
                if self.debug:
                    print("\n⚠️ API recommendation disabled (no API database)")
                self.enable_api_recommendation = False

    def _load_api_database(
        self,
        api_name_path: Optional[Path],
        api_description_path: Optional[Path]
    ) -> Dict[str, Any]:
        print("\n[API Files Status]")
        api_database = {}

        if api_name_path and api_name_path.exists():
            try:
                with open(api_name_path, 'r', encoding='utf-8') as f:
                    content = json.load(f)
                if isinstance(content, dict):
                    api_names = content
                elif isinstance(content, list):
                    api_names = {str(i): name for i, name in enumerate(content)}
                else:
                    api_names = {}

                for api_id, api_name in api_names.items():
                    api_database[api_id] = {'name': api_name, 'description': '', 'category': '', 'tags': []}

                print(f"  ✓ api_name.json: {len(api_names)} APIs loaded")
            except Exception as e:
                print(f"  ✗ api_name.json: Failed to load ({e})")
        else:
            print(f"  ✗ api_name.json: Not found")

        if api_description_path and api_description_path.exists():
            try:
                with open(api_description_path, 'r', encoding='utf-8') as f:
                    api_descriptions = json.load(f)

                if isinstance(api_descriptions, list):
                    api_descriptions = {str(i): item for i, item in enumerate(api_descriptions)}


                loaded_count = 0
                for api_id, item in api_descriptions.items():
                    if isinstance(item, dict):
                        description = str(item.get('description', '') or '')
                        category = str(item.get('category', '') or '')
                        tags = item.get('tags', []) or []
                        if not isinstance(tags, list):
                            tags = [str(tags)]
                    elif isinstance(item, list):
                        description = ' '.join(str(w) for w in item)
                        category, tags = '', []
                    else:
                        description = str(item) if item else ''
                        category, tags = '', []

                    if api_id in api_database:
                        api_database[api_id]['description'] = description
                        api_database[api_id]['category'] = category
                        api_database[api_id]['tags'] = tags
                    else:
                        api_database[api_id] = {
                            'name': api_id, 'description': description,
                            'category': category, 'tags': tags
                        }
                    if description.strip():
                        loaded_count += 1

                print(f"  ✓ api_description.json: {len(api_descriptions)} entries loaded "
                      f"({loaded_count} with non-empty description)")

                if api_database and loaded_count < len(api_database) * 0.5:
                    print(f"  ⚠️ WARNING: only {loaded_count}/{len(api_database)} APIs have a "
                          f"non-empty description after loading. Embeddings for the rest will "
                          f"be built from the bare API name only, which significantly weakens "
                          f"semantic retrieval. Double-check that api_description.json's keys "
                          f"match api_name.json's keys and that the field is really named "
                          f"'description'.")
            except Exception as e:
                print(f"  ✗ api_description.json: Failed to load ({e})")
        else:
            print(f"  ✗ api_description.json: Not found")


        for api_id, info in api_database.items():
            if not info.get('description', '').strip():
                readable_name = re.sub(r'[-_]+', ' ', str(info.get('name', api_id))).strip()
                if readable_name.lower() != str(info.get('name', '')).lower():
                    info['description'] = readable_name

        return api_database



    def _get_scenario_constraints(self, process_description: str) -> Tuple[str, int, int]:

        min_tasks, max_tasks = _estimate_task_range(process_description)
        return "generic", min_tasks, max_tasks

    def _trim_excess_tasks(self, bpmn_result: Dict, max_tasks: int, scenario: str) -> Dict:

        eval_format = bpmn_result.get("evaluation_format", {})
        tasks = eval_format.get("tasks", [])

        current_count = len(tasks)
        if current_count <= max_tasks:
            return bpmn_result

        if self.debug:
            print(f"  ⚠️ Trimming tasks: {current_count} → {max_tasks} (scenario={scenario})")


        trimmed_tasks = tasks[:max_tasks]
        trimmed_ids = {t.get('task_id', t.get('id', '')) for t in trimmed_tasks}


        dependencies = eval_format.get("dependencies", [])
        trimmed_deps = [
            d for d in dependencies
            if d.get('source', d.get('from', '')) in trimmed_ids
            and d.get('target', d.get('to', '')) in trimmed_ids
        ]


        bpmn_result["evaluation_format"]["tasks"] = trimmed_tasks
        bpmn_result["evaluation_format"]["dependencies"] = trimmed_deps


        if "enhanced_info" in bpmn_result:
            bpmn_result["enhanced_info"]["task_count"] = len(trimmed_tasks)
            ei_tasks = bpmn_result["enhanced_info"].get("tasks", [])
            bpmn_result["enhanced_info"]["tasks"] = [
                t for t in ei_tasks
                if t.get('id', '') in trimmed_ids
            ]

        return bpmn_result

    def _generate_bpmn_with_validation(
        self,
        process_description: str,
        max_retries: int = 3,
        validate: bool = True
    ) -> Dict[str, Any]:

        scenario, min_tasks, max_tasks = self._get_scenario_constraints(process_description)

        if self.debug:
            print(f"\n[Scenario Detection] Type: {scenario}, "
                  f"Min={min_tasks}, Max={max_tasks}")

        best_result = None
        best_task_count = 0

        best_score = -999

        def target_score(count):

            if min_tasks <= count <= max_tasks:
                return 100 - abs(count - (min_tasks + max_tasks) // 2)
            elif count < min_tasks:
                return count - min_tasks
            else:
                return max_tasks - count

        for attempt in range(3):
            if self.debug:
                print(f"\n[BPMN Generation Attempt {attempt + 1}/3]")

            if attempt == 0:
                desc_to_use = process_description
            elif attempt == 1:
                desc_to_use = self._augment_description_with_task_requirement(
                    process_description, scenario, min_tasks, max_tasks, level="strong"
                )
            else:
                desc_to_use = self._augment_description_with_task_requirement(
                    process_description, scenario, min_tasks, max_tasks, level="maximum"
                )

            result = self.bpmn_generator.generate(
                process_description=desc_to_use,
                max_retries=max_retries,
                validate=validate
            )


            if result is None:
                if self.debug:
                    print(f"  ⚠️ Attempt {attempt + 1}: bpmn_generator returned None, skipping")
                continue

            task_count = result.get("enhanced_info", {}).get("task_count", 0)

            if self.debug:
                print(f"  → Generated {task_count} tasks "
                      f"(need {min_tasks}~{max_tasks})")

            score = target_score(task_count)

            if score > best_score:
                best_score = score
                best_task_count = task_count
                best_result = result


            if min_tasks <= task_count <= max_tasks:
                if self.debug:
                    print(f"  ✓ Task count in range [{min_tasks}, {max_tasks}]")
                return result


        if best_result is None:
            if self.debug:
                print(f"  ⚠️ All attempts returned None, using fallback")
            best_result = self.bpmn_generator._generate_fallback(process_description)
            best_task_count = best_result.get("enhanced_info", {}).get("task_count", 0)

        if self.debug:
            print(f"\n  → Best result has {best_task_count} tasks")


        if best_task_count > max_tasks:
            if self.debug:
                print(f"  ✂️ Auto-trimming to max_tasks={max_tasks}")
            best_result = self._trim_excess_tasks(best_result, max_tasks, scenario)

        return best_result

    def _augment_description_with_task_requirement(
        self,
        process_description: str,
        scenario: str,
        min_tasks: int,
        max_tasks: int,
        level: str = "strong"
    ) -> str:

        if level == "strong":
            suffix = f"""

[GENERATION REQUIREMENT]
Based on the structure of the process description above (conditions, parallel
activities, sequential steps), generate a BPMN workflow with approximately
{min_tasks}-{max_tasks} tasks. Split each distinct activity mentioned in the
text into its own task; do not merge multiple described actions into one task,
and do not invent activities that are not implied by the text.
Every exclusiveGateway and inclusiveGateway MUST have "has_join": true.
"""
        else:  # maximum
            suffix = f"""

[GENERATION REQUIREMENT - STRICT]
Re-read the process description carefully. Generate a BPMN workflow with
{min_tasks}-{max_tasks} tasks total:
- If the text describes conditional branches, model them as exclusiveGateway
  or inclusiveGateway (with has_join: true) around exactly the branches
  described in the text.
- If the text describes activities happening concurrently/independently of
  order, model them as a parallelGateway.
- Do not add steps, roles, or checks that are not stated or clearly implied
  by the text. Do not omit a step that is explicitly stated.
Stay within the {min_tasks}-{max_tasks} task range.
"""

        return process_description + suffix


    def generate(
        self,
        process_description: str,
        max_retries: int = 3,
        validate: bool = True
    ) -> Dict[str, Any]:

        if self.debug:
            print("\n" + "="*70)
            print("Processing User Request")
            print("="*70)
            print(f"Input: {process_description[:200]}...")


        bpmn_result = self._generate_bpmn_with_validation(
            process_description=process_description,
            max_retries=max_retries,
            validate=validate
        )


        if bpmn_result is None:
            if self.debug:
                print("  ⚠️ bpmn_result is None after all attempts, using emergency fallback")
            bpmn_result = self.bpmn_generator._generate_fallback(process_description)


        api_bindings = {}

        if self.enable_api_recommendation and self.api_recommender:
            tasks = self._extract_tasks_from_bpmn(bpmn_result)

            if self.debug:
                print(f"\n✓ Extracted {len(tasks)} tasks from workflow")

            if tasks:
                api_bindings = self.api_recommender.recommend_for_tasks(
                    tasks,
                    process_description
                )


        result = {
            "bpmn_workflow": bpmn_result.get("bpmn_workflow", {}),
            "evaluation_format": bpmn_result.get("evaluation_format", {}),
            "enhanced_info": bpmn_result.get("enhanced_info", {}),
            "validation": bpmn_result.get("validation", {"is_valid": True}),
            "api_bindings": api_bindings,
            "api_recommendation_stats": self.api_recommender.stats if self.api_recommender else {},
            "complete_output": self._merge_api_bindings(
                bpmn_result.get("evaluation_format", {}),
                api_bindings
            ),
            "metadata": {
                "generator": "CompleteWorkflowGenerator v3.7",
                "timestamp": bpmn_result.get("metadata", {}).get("timestamp", ""),
                "input_length": len(process_description),
                "has_api_recommendation": len(api_bindings) > 0
            }
        }

        if self.debug:
            print("\n" + "="*70)
            print("Generation Complete")
            print("="*70)
            print(f"\n[BPMN Statistics]")
            print(f"  Tasks: {result['enhanced_info'].get('task_count', 0)}")
            print(f"  Gateways: {result['enhanced_info'].get('gateway_count', 0)}")
            print(f"\n[API Recommendations]")
            print(f"  Total bindings: {len(api_bindings)}")
            unique = len(set(b.get('api_name', '') for b in api_bindings.values()))
            print(f"  Unique APIs: {unique}")

        return result

    def _extract_tasks_from_bpmn(self, bpmn_result: Dict) -> List[Dict]:

        if bpmn_result is None:
            return []

        tasks = []

        eval_format = bpmn_result.get("evaluation_format", {})
        if "tasks" in eval_format:
            for task in eval_format["tasks"]:
                tasks.append({
                    'task_id': task.get('task_id', ''),
                    'id': task.get('task_id', ''),
                    'name': task.get('name', ''),
                    'description': task.get('name', ''),
                    'type': task.get('type', 'task')
                })
        else:
            bpmn_workflow = bpmn_result.get("bpmn_workflow", {})
            process = bpmn_workflow.get("process", [])

            task_counter = 0
            for element in process:
                if isinstance(element, dict) and element.get('type') in [
                    'userTask', 'serviceTask', 'sendTask', 'receiveTask',
                    'manualTask', 'businessRuleTask', 'scriptTask', 'task'
                ]:
                    task_counter += 1
                    task_id = element.get('id', f't{task_counter}')
                    tasks.append({
                        'task_id': task_id,
                        'id': task_id,
                        'name': element.get('label', f'Task {task_counter}'),
                        'description': element.get('label', ''),
                        'type': element.get('type', 'task')
                    })

        return tasks

    def _merge_api_bindings(
        self,
        evaluation_format: Dict,
        api_bindings: Dict
    ) -> Dict:

        merged = evaluation_format.copy()

        if "tasks" in merged:
            for task in merged["tasks"]:
                task_id = task.get('task_id', '')
                if task_id in api_bindings:
                    task['api_recommendation'] = api_bindings[task_id]

        return merged
