import re
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TaskEnhancer:

    VERB_MAPPING = {

        'clarify':    'clarify',
        'clarifies':  'clarify',
        'determine':  'determine',
        'determines': 'determine',
        'decide':     'decide',
        'decides':    'decide',


        'assess':     'assess',
        'assesses':   'assess',
        'evaluate':   'evaluate',
        'evaluates':  'evaluate',
        'check':      'check',
        'checks':     'check',
        'verify':     'verify',
        'verifies':   'verify',
        'validate':   'validate',
        'validates':  'validate',
        'inspect':    'inspect',
        'inspects':   'inspect',
        'review':     'review',
        'reviews':    'review',


        'calculate':  'calculate',
        'calculates': 'calculate',
        'compute':    'compute',
        'computes':   'compute',
        'score':      'score',
        'scores':     'score',


        'approve':    'approve',
        'approves':   'approve',
        'approved':   'approve',
        'reject':     'reject',
        'rejects':    'reject',
        'rejected':   'reject',
        'confirm':    'confirm',
        'confirms':   'confirm',


        'submit':     'submit',
        'submits':    'submit',
        'receive':    'receive',
        'receives':   'receive',
        'register':   'register',
        'registers':  'register',


        'assign':     'assign',
        'assigns':    'assign',
        'schedule':   'schedule',
        'schedules':  'schedule',
        'allocate':   'allocate',
        'allocates':  'allocate',


        'monitor':    'monitor',
        'monitors':   'monitor',
        'track':      'track',
        'tracks':     'track',
        'update':     'update',
        'updates':    'update',


        'cancel':     'cancel',
        'cancels':    'cancel',
        'terminate':  'terminate',
        'terminates': 'terminate',


        'write':      'write',
        'writes':     'write',
        'written':    'write',
        'create':     'create',
        'creates':    'create',
        'generate':   'generate',
        'generates':  'generate',


        'prepare':    'prepare',
        'prepares':   'prepare',
        'prepared':   'prepare',
        'process':    'process',
        'processes':  'process',
        'handle':     'handle',
        'handles':    'handle',
        'perform':    'perform',
        'performs':   'perform',
        'execute':    'execute',
        'executes':   'execute',


        'invite':     'request',
        'invites':    'request',
        'request':    'request',
        'requests':   'request',
        'ask':        'request',
        'asks':       'request',
        'notify':     'notify',
        'notifies':   'notify',
        'inform':     'inform',
        'informs':    'inform',
        'send':       'send',
        'sends':      'send',


        'select':     'select',
        'selects':    'select',
        'choose':     'choose',
        'chooses':    'choose',
        'pick':       'select',
        'picks':      'select',


        'take':       'obtain',
        'takes':      'obtain',
        'taken':      'obtain',
        'get':        'get',
        'gets':       'get',
        'obtain':     'obtain',
        'obtains':    'obtain',
        'retrieve':   'retrieve',
        'retrieves':  'retrieve',
        'fetch':      'retrieve',
        'fetches':    'retrieve',


        'package':    'package',
        'packages':   'package',
        'packaged':   'package',
        'ship':       'ship',
        'ships':      'ship',
        'deliver':    'deliver',
        'delivers':   'deliver',


        'make':       'make',
        'makes':      'make',
        'do':         'perform',
        'does':       'perform',
        'arrange':    'arrange',
        'arranges':   'arrange',
        'complete':   'complete',
        'completes':  'complete',
        'finalize':   'finalize',
        'finalizes':  'finalize',
    }


    _FALLBACK_DOMAIN_KEYWORDS = {

        'shipping':    'logistics carrier API',
        'shipment':    'shipment tracking API',
        'ship':        'transport delivery service',
        'dispatch':    'logistics coordination service',
        'delivery':    'delivery routing API',
        'carrier':     'carrier selection API',
        'logistics':   'logistics management API',
        'freight':     'freight booking service',
        'transport':   'transportation API',
        'courier':     'courier dispatch API',
        'pickup':      'pickup scheduling API',
        'route':       'route optimization API',


        'label':       'label generation API',
        'package':     'packaging fulfillment service',
        'parcel':      'parcel handling API',
        'packing':     'packaging automation service',
        'barcode':     'barcode generation API',
        'qr':          'QR code generation API',


        'insurance':   'insurance coverage API',
        'insure':      'insurance purchase service',
        'coverage':    'coverage calculation API',
        'premium':     'premium calculation API',
        'policy':      'insurance policy API',


        'warehouse':   'warehouse management API',
        'storage':     'inventory storage API',
        'inventory':   'inventory control API',
        'stock':       'stock management API',
        'picking':     'warehouse picking API',
        'goods':       'goods handling API',


        'offer':       'quote comparison API',
        'quote':       'rate pricing API',
        'bid':         'bidding service API',
        'price':       'pricing calculation API',
        'cost':        'cost estimation API',
        'rate':        'rate lookup API',


        'order':       'order management API',
        'purchase':    'purchase transaction API',
        'checkout':    'checkout processing API',
        'cart':        'shopping cart API',
        'payment':     'payment gateway API',
        'invoice':     'invoice generation API',
        'receipt':     'receipt API',


        'customer':    'customer management API',
        'client':      'client service API',
        'user':        'user management API',
        'account':     'account management API',
        'profile':     'user profile API',


        'recourse':    'complaint handling API',
        'claim':       'claim processing API',
        'complaint':   'complaint resolution API',
        'dispute':     'dispute management API',
        'refund':      'refund processing API',
        'compensation':'compensation calculation API',


        'credit':      'credit assessment API',
        'score':       'scoring engine API',
        'scoring':     'credit scoring API',
        'rating':      'credit rating API',
        'loan':        'loan processing API',
        'mortgage':    'mortgage evaluation API',
        'bank':        'banking service API',
        'finance':     'financial service API',
        'risk':        'risk assessment API',
        'fraud':       'fraud detection API',
        'kyc':         'KYC verification API',
        'aml':         'AML compliance API',
        'income':      'income verification API',
        'asset':       'asset valuation API',
        'application': 'application processing API',
        'approve':     'approval workflow API',
        'reject':      'rejection notification API',


        'restaurant':  'restaurant management API',
        'food':        'food ordering API',
        'meal':        'meal service API',
        'dish':        'dish selection API',
        'menu':        'menu catalog API',
        'beverage':    'beverage ordering API',
        'drink':       'drink selection API',
        'dessert':     'dessert ordering API',
        'tray':        'tray management service',
        'cafeteria':   'cafeteria management API',
        'station':     'service station API',
        'buffet':      'buffet management API',
        'cook':        'kitchen preparation API',
        'prepare':     'preparation service API',
        'serve':       'service delivery API',
        'seat':        'seating management API',
        'table':       'table reservation API',


        'workflow':    'workflow engine API',
        'approval':    'approval management API',
        'review':      'review process API',
        'audit':       'audit trail API',
        'compliance':  'compliance check API',
        'regulation':  'regulatory API',
        'document':    'document management API',
        'form':        'form processing API',
        'report':      'reporting API',
        'notification': 'notification service API',
        'alert':       'alert management API',
        'email':       'email service API',
        'sms':         'SMS notification API',
    }


    ROLE_WORDS = [
        'secretary',
        'the secretary',
        'manager',
        'the manager',
        'department head',
        'logistics department head',
        'logistics head',
        'warehousemen',
        'the warehousemen',
        'warehouse staff',
        'staff',
        'worker',
        'employee',
        'clerk',
        'agent',
        'operator',
        'head',
        'officer',
        'supervisor',
        'coordinator',
        'director',
        'analyst',
    ]


    PASSIVE_PATTERNS = [
        (r'is\s+(\w+ed)\s+by',  r'\1'),
        (r'are\s+(\w+ed)\s+by', r'\1'),
        (r'was\s+(\w+ed)\s+by', r'\1'),
        (r'were\s+(\w+ed)\s+by',r'\1'),
        (r'is\s+(\w+ed)',        r'\1'),
        (r'are\s+(\w+ed)',       r'\1'),
        (r'was\s+(\w+ed)',       r'\1'),
        (r'were\s+(\w+ed)',      r'\1'),
    ]


    _DEFAULT_AUTO_KEYWORDS_CANDIDATES = [
        "domain_keywords_auto.json",
        "data/domain_keywords_auto.json",
    ]

    def __init__(self, domain_keywords_path: Optional[str] = None,
                 require_evidence_for_single_word: bool = False):

        self.require_evidence_for_single_word = require_evidence_for_single_word
        (self.DOMAIN_KEYWORDS,
         self._keyword_source,
         self.CATEGORY_TOP_WORDS,
         self.CATEGORY_IS_GENERIC) = self._load_domain_keywords(domain_keywords_path)

        self._keyword_source_is_auto: bool = self._keyword_source.startswith("auto-mined")

        self._category_stem_tokens: Dict[str, List[str]] = {
            cat: [self._stem(t) for t in re.split(r"[-_\s]+", cat) if t]
            for cat in self.DOMAIN_KEYWORDS
        }

        self._category_topword_stems: Dict[str, set] = {
            cat: {
                self._stem(w) for w in self.CATEGORY_TOP_WORDS.get(cat, [])
                if self._stem(w) not in set(self._category_stem_tokens.get(cat, []))
            }
            for cat in self.DOMAIN_KEYWORDS
        }


        stem_category_df: Dict[str, int] = {}
        for cat, words in self.CATEGORY_TOP_WORDS.items():
            for w in {self._stem(x) for x in words}:
                stem_category_df[w] = stem_category_df.get(w, 0) + 1

        num_categories = max(1, len(self.CATEGORY_TOP_WORDS))

        self._generic_word_df_threshold = max(20, math.ceil(0.07 * num_categories))
        self._cross_category_generic_stems: set = {
            stem for stem, df in stem_category_df.items()
            if df >= self._generic_word_df_threshold
        }


        self._category_evidence_stems: Dict[str, set] = {
            cat: stems - self._cross_category_generic_stems
            for cat, stems in self._category_topword_stems.items()
        }

        print(f"[TaskEnhancer] 领域关键词表来源: {self._keyword_source}"
              f"（共 {len(self.DOMAIN_KEYWORDS)} 个领域词条，"
              f"跨类目通用词 {len(self._cross_category_generic_stems)} 个，"
              f"阈值=出现在 >= {self._generic_word_df_threshold} 个类目的 top_words 中）")

    @staticmethod
    def _stem(word: str) -> str:

        w = word.lower()
        if len(w) > 4 and w.endswith("ies"):
            return w[:-3] + "y"
        if len(w) > 4 and w.endswith(("ches", "shes", "xes", "ses")):
            return w[:-2]
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        return w

    def _load_domain_keywords(
        self, domain_keywords_path: Optional[str]
    ) -> Tuple[Dict[str, str], str, Dict[str, List[str]], Dict[str, bool]]:

        candidate_paths: List[Path] = []
        if domain_keywords_path:
            candidate_paths.append(Path(domain_keywords_path))
        else:
            here = Path(__file__).resolve().parent
            candidate_paths.extend(here / p for p in self._DEFAULT_AUTO_KEYWORDS_CANDIDATES)

        for path in candidate_paths:
            if path.is_file():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    # 自动挖掘脚本输出格式: {category: {"injection": ..., "top_words": [...], ...}}
                    flat = {cat: info["injection"] for cat, info in raw.items()}
                    top_words = {
                        cat: list(info.get("top_words", []))
                        for cat, info in raw.items()
                    }
                    is_generic = {
                        cat: bool(info.get("is_generic", False))
                        for cat, info in raw.items()
                    }
                    if flat:
                        return flat, f"auto-mined ({path})", top_words, is_generic
                except Exception as e:  # noqa: BLE001
                    print(f"[TaskEnhancer] 警告：加载 {path} 失败（{e}），继续尝试其他来源")

        print("[TaskEnhancer] 警告：未找到自动挖掘词表 domain_keywords_auto.json，"
              "回退到 v1.1 手写词表 _FALLBACK_DOMAIN_KEYWORDS。"
              "如需使用自动挖掘版，请先运行 mine_domain_keywords.py 生成该文件。")

        fallback_top_words = {
            cat: [w for w in re.findall(r"[a-zA-Z]+", phrase.lower())
                  if w not in {"api", cat}]
            for cat, phrase in self._FALLBACK_DOMAIN_KEYWORDS.items()
        }
        fallback_is_generic = {cat: False for cat in self._FALLBACK_DOMAIN_KEYWORDS}
        return (dict(self._FALLBACK_DOMAIN_KEYWORDS), "fallback (hand-written v1.1)",
                fallback_top_words, fallback_is_generic)

    def match_categories(self, text: str) -> List[str]:

        text_tokens = {self._stem(t) for t in re.findall(r"[a-zA-Z]+", text.lower())}


        _PARTIAL_MATCH_MIN_RATIO = 0.5

        matched = []  # List[Tuple[cat, is_full_match]]
        for cat, stem_subtokens in self._category_stem_tokens.items():
            if not stem_subtokens:
                continue

            hit_count = sum(1 for st in stem_subtokens if st in text_tokens)
            is_full = hit_count == len(stem_subtokens)

            if len(stem_subtokens) == 1:
                if not is_full:
                    continue


                if self._keyword_source_is_auto and self.require_evidence_for_single_word:

                    evidence_stems = self._category_evidence_stems.get(cat, set())
                    if not (evidence_stems & text_tokens):
                        continue

                matched.append((cat, True))
                continue


            if is_full:
                matched.append((cat, True))
                continue


            if hit_count / len(stem_subtokens) >= _PARTIAL_MATCH_MIN_RATIO:
                evidence_stems = self._category_evidence_stems.get(cat, set())
                if evidence_stems and (evidence_stems & text_tokens):
                    matched.append((cat, False))

        matched.sort(
            key=lambda item: (
                item[1],
                not self.CATEGORY_IS_GENERIC.get(item[0], False),
                len(self._category_stem_tokens[item[0]]),
                len(item[0]),
            ),
            reverse=True,
        )
        return [cat for cat, _ in matched]


    def enhance(self, task: Dict) -> Dict:

        name = task.get('name', '')
        desc = task.get('description', '')

        task['original_name']        = name
        task['original_description'] = desc

        name = self._remove_role_words(name)
        desc = self._remove_role_words(desc)

        name = self._passive_to_active(name)
        desc = self._passive_to_active(desc)

        name = self._standardize_verbs(name)

        name = self._clean_redundancy(name)
        desc = self._clean_redundancy(desc)


        desc = self._inject_api_keywords(name, desc)

        name = self._clean_format(name)
        desc = self._clean_format(desc)

        if name:
            name = name[0].upper() + name[1:] if len(name) > 1 else name.upper()

        task['name']        = name
        task['description'] = desc
        task['enhanced']    = True

        return task

    def _remove_role_words(self, text: str) -> str:

        text_lower = text.lower()
        for role in self.ROLE_WORDS:
            text_lower = text_lower.replace(f' by {role}', '')
            text_lower = text_lower.replace(f' by the {role}', '')
            text_lower = text_lower.replace(f'{role} ', '')
            text_lower = text_lower.replace(f'the {role} ', '')
            text_lower = text_lower.replace(f' {role} ', ' ')
        return text_lower

    def _passive_to_active(self, text: str) -> str:

        text = re.sub(r'\b(is|are|was|were)\s+(\w+ed)\s+by\b', r'\2', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(is|are|was|were)\s+(\w+en)\s+by\b', r'\2', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(is|are|was|were)\s+(\w+ed)\b',      r'\2', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(is|are|was|were)\s+(\w+en)\b',      r'\2', text, flags=re.IGNORECASE)
        return text

    def _standardize_verbs(self, text: str) -> str:

        words = text.split()
        for i, word in enumerate(words):
            word_clean = word.strip('.,;:!?').lower()
            if word_clean in self.VERB_MAPPING:
                words[i] = self.VERB_MAPPING[word_clean]
                break
        return ' '.join(words)

    def _clean_redundancy(self, text: str) -> str:

        redundancy_patterns = [
            r'\b(who will do|who does|who should)\b',
            r'\b(it is necessary to|necessary to)\b',
            r'\bin order to\b',
            r'\bshall be\b',
            r'\bwill be\b',
            r'\bcan be\b',
            r'\bmust be\b',
            r'\bshould be\b',
            r'\bneeds to be\b',
            r'\bhas to be\b',
        ]
        for pattern in redundancy_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        return text


    def _inject_api_keywords(self, name: str, desc: str) -> str:

        combined_text = f"{name} {desc}"
        matched_categories = self.match_categories(combined_text)

        if not matched_categories:
            return desc


        seen_terms: set = set()
        selected_terms: List[str] = []
        for cat in matched_categories:
            api_term = self.DOMAIN_KEYWORDS[cat]

            base = api_term.split()[0].lower()
            if base not in seen_terms and len(selected_terms) < 3:
                seen_terms.add(base)
                selected_terms.append(api_term)

        if not selected_terms:
            return desc


        combined = " | ".join(selected_terms)
        desc = desc.rstrip('.').rstrip() + f' ({combined}).'
        return desc

    def _combine_api_terms(self, terms: List[str]) -> str:

        if not terms:
            return ""
        if len(terms) == 1:
            return f"({terms[0]})"
        return "(" + " | ".join(terms) + ")"

    def _clean_format(self, text: str) -> str:

        text = ' '.join(text.split())
        text = re.sub(r'^(the|a|an)\s+',     '', text, flags=re.IGNORECASE)
        text = re.sub(r'^(and|or|but|so|then)\s+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\b(the|a|an)\s+(the|a|an)\b', r'\1', text, flags=re.IGNORECASE)
        return text.strip()

    def enhance_batch(self, tasks: List[Dict]) -> List[Dict]:

        enhanced_tasks = []
        for task in tasks:
            task_copy     = task.copy()
            enhanced_task = self.enhance(task_copy)
            enhanced_tasks.append(enhanced_task)
        return enhanced_tasks

    def get_enhancement_summary(self, original_tasks: List[Dict], enhanced_tasks: List[Dict]) -> Dict:

        total = len(original_tasks)
        enhanced_count       = 0
        role_removed_count   = 0
        verb_changed_count   = 0
        keyword_injected_count = 0

        for orig, enhanced in zip(original_tasks, enhanced_tasks):
            orig_name     = orig.get('name', '')
            enhanced_name = enhanced.get('name', '')
            orig_desc     = orig.get('description', '')
            enhanced_desc = enhanced.get('description', '')

            if orig_name != enhanced_name or orig_desc != enhanced_desc:
                enhanced_count += 1

            if any(role in orig_name.lower() or role in orig_desc.lower()
                   for role in self.ROLE_WORDS):
                role_removed_count += 1

            orig_first     = orig_name.split()[0].lower()     if orig_name.split()     else ''
            enhanced_first = enhanced_name.split()[0].lower() if enhanced_name.split() else ''
            if orig_first != enhanced_first:
                verb_changed_count += 1

            if '(' in enhanced_desc and ')' in enhanced_desc:
                keyword_injected_count += 1

        return {
            'total_tasks':           total,
            'enhanced_count':        enhanced_count,
            'role_removed_count':    role_removed_count,
            'verb_changed_count':    verb_changed_count,
            'keyword_injected_count': keyword_injected_count,
            'enhancement_rate':      enhanced_count / total if total > 0 else 0
        }



if __name__ == "__main__":
    test_tasks = [
        {
            'task_id': 'task_1',
            'name': 'the secretary clarifies who will do the shipping',
            'description': 'The secretary determines which carrier should handle the shipment'
        },
        {
            'task_id': 'task_2',
            'name': 'a package label is written by the secretary',
            'description': 'Create and print the shipping label for the parcel'
        },
        {
            'task_id': 'task_3',
            'name': 'the warehousemen package goods',
            'description': 'Goods are packaged by warehouse staff for delivery'
        },
        {
            'task_id': 'task_4',
            'name': 'parcel insurance taken by logistics department head',
            'description': 'Purchase insurance coverage if necessary'
        },
        {
            'task_id': 'task_5',
            'name': 'calculate credit score for loan application',
            'description': 'Run credit scoring model on applicant data'
        },
        {
            'task_id': 'task_6',
            'name': 'customer selects food from restaurant menu',
            'description': 'Customer browses the menu and selects a dish'
        },
    ]

    enhancer = TaskEnhancer()

    print("=" * 80)
    print("TaskEnhancer v1.6 测试")
    print("=" * 80)

    for task in test_tasks:
        print(f"\n[{task['task_id']}]")
        print(f"  原始名称: {task['name']}")
        enhanced = enhancer.enhance(task.copy())
        print(f"  增强名称: {enhanced['name']}")
        print(f"  原始描述: {task['description']}")
        print(f"  增强描述: {enhanced['description']}")

    enhanced_batch = enhancer.enhance_batch(test_tasks)
    summary = enhancer.get_enhancement_summary(test_tasks, enhanced_batch)

    print("\n" + "=" * 80)
    print("增强效果摘要")
    print("=" * 80)
    print(f"  总任务数:     {summary['total_tasks']}")
    print(f"  增强任务数:   {summary['enhanced_count']}")
    print(f"  移除角色词:   {summary['role_removed_count']}")
    print(f"  动词标准化:   {summary['verb_changed_count']}")
    print(f"  关键词注入:   {summary['keyword_injected_count']}")
    print(f"  增强率:       {summary['enhancement_rate']:.1%}")
