import json
import math
import re
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


class TaskEnhancer:


    STEPS = ("remove_role", "passive_to_active", "verb_standardize", "inject_keywords")


    ROLE_LEXICON = [

        "secretary", "manager", "department head", "head", "director",
        "supervisor", "coordinator", "officer", "clerk", "employee", "staff",
        "worker", "operator", "agent", "assistant", "administrator",
        "receptionist",

        "logistics department head", "logistics head", "warehouse staff",
        "warehouseman", "warehousemen",

        "analyst", "accountant", "cashier", "teller", "underwriter",
        "loan officer", "credit officer",

        "customer service representative", "service representative",
        "representative",
    ]


    _ROLE_FILLER_PATTERN = (
        r"\bwho\s+(?:will|shall|should|does|do|is\s+going\s+to)"
        r"(?:\s+(?:do|handle|be\s+responsible\s+for))?\b"
    )


    _PASSIVE_RE = re.compile(
        r"^(?P<obj>.+?)\s+(?:is|are|was|were)\s+(?:being\s+)?(?P<pp>[a-z]+)(?P<rest>(?:\s+.*)?)$"
    )
    _SUBORDINATORS = frozenset({
        "if", "when", "once", "after", "before", "while", "unless",
        "whether", "because", "since", "as", "until",
    })
    _DETERMINERS = frozenset({"the", "a", "an", "this", "that", "these", "those"})
    _AGENT_STOP_WORDS = frozenset({
        "for", "to", "in", "on", "at", "with", "from", "of", "and", "or",
        "if", "when", "before", "after", "then", "using", "via", "into",
    })


    IRREGULAR_VERB_FORMS = {
        "does": "do", "did": "do", "done": "do",
        "takes": "take", "took": "take", "taken": "take",
        "makes": "make", "made": "make",
        "gets": "get", "got": "get", "gotten": "get",
        "writes": "write", "wrote": "write", "written": "write",
        "sends": "send", "sent": "send",
        "chooses": "choose", "chose": "choose", "chosen": "choose",
        "gives": "give", "gave": "give", "given": "give",
        "shows": "show", "showed": "show", "shown": "show",
        "holds": "hold", "held": "hold",
        "pays": "pay", "paid": "pay",
        "keeps": "keep", "kept": "keep",
        "buys": "buy", "bought": "buy",
        "finds": "find", "found": "find",
        "tells": "tell", "told": "tell",
        "builds": "build", "built": "build",
    }


    BUSINESS_VERBS = frozenset({
        "clarify", "determine", "decide", "assess", "evaluate", "check",
        "verify", "validate", "inspect", "review", "calculate", "compute",
        "score", "approve", "reject", "confirm", "submit", "receive",
        "register", "assign", "schedule", "allocate", "monitor", "track",
        "update", "cancel", "terminate", "write", "create", "generate",
        "prepare", "process", "handle", "perform", "execute", "request",
        "notify", "send", "select", "obtain", "retrieve", "package", "ship",
        "deliver", "arrange", "complete", "finalize", "print", "issue",
        "record", "store", "pay", "charge", "book", "reserve", "deposit",
        "transfer", "authorize", "publish", "collect", "sign", "archive",
        "close", "open", "forward", "return", "refund", "claim", "file",
        "escalate", "resolve", "assemble", "load", "unload", "pack", "label",
        "inform", "offer", "quote", "invoice", "insure", "purchase", "order",
        "provide", "display", "install", "deploy", "enroll", "search",
    })


    VERB_NORMALIZATION = {
        "invite": "request", "ask": "request",
        "take": "obtain", "acquire": "obtain", "procure": "obtain",
        "collect": "obtain", "gather": "obtain",
        "make": "create", "build": "create", "construct": "create",
        "produce": "generate",
        "pick": "select", "choose": "select",
        "do": "perform", "conduct": "perform",
        "fetch": "retrieve", "pull": "retrieve", "grab": "retrieve",
        "find": "search",
        "inform": "notify", "tell": "notify", "alert": "notify", "remind": "notify",
        "modify": "update", "change": "update", "edit": "update",
        "revise": "update", "amend": "update",
        "remove": "delete", "erase": "delete", "discard": "delete",
        "forward": "send", "transmit": "send",
        "arrange": "schedule", "organize": "schedule",
        "examine": "inspect",
        "buy": "purchase",
        "give": "provide",
        "show": "display",
    }


    _DEFAULT_TABLE_CANDIDATES = [
        "domain_keywords_auto.json",
        "data/domain_keywords_auto.json",
    ]
    MAX_INJECTED_PHRASES = 3
    _PARTIAL_MATCH_MIN_RATIO = 0.5

    def __init__(
        self,
        domain_keywords_path: Optional[str] = None,
        domain_table: Optional[Dict[str, Dict]] = None,
        require_evidence_for_single_word: bool = False,
        enabled_steps: Optional[Iterable[str]] = None,
        verbose: bool = True,
    ):

        self.require_evidence_for_single_word = require_evidence_for_single_word
        steps = tuple(enabled_steps) if enabled_steps is not None else self.STEPS
        unknown = set(steps) - set(self.STEPS)
        if unknown:
            raise ValueError("Unknown enhancement steps: %s" % sorted(unknown))
        self.enabled_steps = set(steps)

        (self.DOMAIN_KEYWORDS,
         self.CATEGORY_TOP_WORDS,
         self.CATEGORY_IS_GENERIC,
         self._table_source) = self._load_domain_table(domain_keywords_path, domain_table)


        roles = sorted({r.lower() for r in self.ROLE_LEXICON}, key=len, reverse=True)
        self._role_re = re.compile(
            r"\b(?:by\s+)?(?:(?:the|a|an)\s+)?(?:%s)s?\b" % "|".join(re.escape(r) for r in roles)
        )
        self._role_filler_re = re.compile(self._ROLE_FILLER_PATTERN)

        self._known_verbs: Set[str] = (
            set(self.BUSINESS_VERBS)
            | set(self.VERB_NORMALIZATION)
            | set(self.VERB_NORMALIZATION.values())
        )


        self._category_stem_tokens: Dict[str, List[str]] = {
            cat: [self._stem(t) for t in re.split(r"[-_\s]+", cat) if t]
            for cat in self.DOMAIN_KEYWORDS
        }


        topword_stems: Dict[str, Set[str]] = {
            cat: {
                self._stem(w) for w in self.CATEGORY_TOP_WORDS.get(cat, [])
                if self._stem(w) not in set(self._category_stem_tokens.get(cat, []))
            }
            for cat in self.DOMAIN_KEYWORDS
        }


        stem_category_df: Dict[str, int] = {}
        for words in self.CATEGORY_TOP_WORDS.values():
            for stem in {self._stem(w) for w in words}:
                stem_category_df[stem] = stem_category_df.get(stem, 0) + 1
        num_categories = max(1, len(self.CATEGORY_TOP_WORDS))
        self._generic_df_threshold = max(20, math.ceil(0.07 * num_categories))
        self._cross_category_generic_stems: Set[str] = {
            s for s, df in stem_category_df.items() if df >= self._generic_df_threshold
        }
        self._category_evidence_stems: Dict[str, Set[str]] = {
            cat: stems - self._cross_category_generic_stems
            for cat, stems in topword_stems.items()
        }

        if verbose:
            print("[TaskEnhancer] domain table: %s (%d categories)"
                  % (self._table_source, len(self.DOMAIN_KEYWORDS)))


    def _load_domain_table(
        self, path: Optional[str], table: Optional[Dict[str, Dict]]
    ) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, bool], str]:

        raw: Optional[Dict[str, Dict]] = None
        source = "none"

        if table is not None:
            raw, source = table, "provided table"
        else:
            if path:
                candidates = [Path(path)]
            else:
                here = Path(__file__).resolve().parent
                candidates = [here / p for p in self._DEFAULT_TABLE_CANDIDATES]
            for candidate in candidates:
                if not candidate.is_file():
                    continue
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        raw = json.load(f)
                    source = str(candidate)
                    break
                except (OSError, json.JSONDecodeError) as exc:
                    warnings.warn("Failed to load %s: %s" % (candidate, exc))

        if not raw:
            warnings.warn(
                "No mined domain table found; keyword injection is disabled. "
                "Run the offline mining script on the API repository to build it."
            )
            return {}, {}, {}, "none"

        keywords = {cat: info["injection"] for cat, info in raw.items()}
        top_words = {cat: list(info.get("top_words", [])) for cat, info in raw.items()}
        is_generic = {cat: bool(info.get("is_generic", False)) for cat, info in raw.items()}
        return keywords, top_words, is_generic, source



    def enhance_description(self, description: str) -> str:

        return self._run_cascade(description)[0]

    def enhance(self, task: Dict) -> Dict:

        original = task.get("description") or task.get("name", "")
        enhanced, trace = self._run_cascade(original)

        task["original_description"] = original
        task["description"] = enhanced
        task["enhanced"] = True
        task["enhancement_trace"] = trace
        return task

    def enhance_batch(self, tasks: List[Dict]) -> List[Dict]:

        return [self.enhance(task.copy()) for task in tasks]

    def get_enhancement_summary(self, enhanced_tasks: List[Dict]) -> Dict:

        total = len(enhanced_tasks)
        traces = [t.get("enhancement_trace", {}) for t in enhanced_tasks]
        changed = sum(
            1 for t in enhanced_tasks
            if t.get("description") != t.get("original_description")
        )
        return {
            "total_tasks": total,
            "enhanced_count": changed,
            "role_removed_count": sum(1 for tr in traces if tr.get("role_removed")),
            "passive_converted_count": sum(1 for tr in traces if tr.get("passive_converted")),
            "verb_standardized_count": sum(1 for tr in traces if tr.get("verb_standardized")),
            "keyword_injected_count": sum(1 for tr in traces if tr.get("injected_phrases")),
            "enhancement_rate": changed / total if total else 0.0,
        }



    def _run_cascade(self, description: str) -> Tuple[str, Dict]:
        text = " ".join((description or "").lower().split())
        trace = {
            "role_removed": False,
            "passive_converted": False,
            "verb_standardized": False,
            "injected_phrases": [],
        }

        if "remove_role" in self.enabled_steps:
            new = self._remove_role_words(text)
            trace["role_removed"] = new != text
            text = new

        if "passive_to_active" in self.enabled_steps:
            new = self._passive_to_active(text)
            trace["passive_converted"] = new != text
            text = new

        if "verb_standardize" in self.enabled_steps:
            new = self._standardize_verbs(text)
            trace["verb_standardized"] = new != text
            text = new

        if "inject_keywords" in self.enabled_steps:
            text, phrases = self._inject_api_keywords(text)
            trace["injected_phrases"] = phrases

        return text, trace



    def _remove_role_words(self, text: str) -> str:
        text = self._role_re.sub(" ", text)
        text = self._role_filler_re.sub(" ", text)
        return " ".join(text.split())



    def _passive_to_active(self, text: str) -> str:
        clauses = [c for c in re.split(r"(?<=[.;])\s+", text.strip()) if c]
        return " ".join(self._convert_clause(c) for c in clauses)

    def _convert_clause(self, clause: str) -> str:
        trailing = ""
        punct = re.search(r"[.;]+$", clause)
        if punct:
            trailing, clause = punct.group(0), clause[:punct.start()]

        match = self._PASSIVE_RE.match(clause)
        if not match:
            return clause + trailing

        obj_tokens = match.group("obj").split()

        if (not obj_tokens
                or any(t in self._SUBORDINATORS for t in obj_tokens)
                or self._to_base_form(obj_tokens[0]) is not None):
            return clause + trailing

        base = self._participle_to_base(match.group("pp"))
        if base is None:
            return clause + trailing

        while obj_tokens and obj_tokens[0] in self._DETERMINERS:
            obj_tokens = obj_tokens[1:]
        if not obj_tokens:
            return clause + trailing

        rest = self._drop_agent_phrase(match.group("rest").split())
        converted = " ".join([base] + obj_tokens + rest)
        return converted + trailing

    def _drop_agent_phrase(self, tokens: List[str]) -> List[str]:

        if not tokens or tokens[0] != "by":
            return tokens
        i = 1
        if i < len(tokens) and tokens[i] in self._DETERMINERS:
            i += 1
        consumed = 0
        while i < len(tokens) and consumed < 2 and tokens[i] not in self._AGENT_STOP_WORDS:
            i += 1
            consumed += 1
        return tokens[i:]

    def _participle_to_base(self, word: str) -> Optional[str]:

        if word in self.IRREGULAR_VERB_FORMS:
            return self.IRREGULAR_VERB_FORMS[word]
        if not word.endswith("ed") or len(word) < 4:
            return None
        if word.endswith("ied"):
            return word[:-3] + "y"
        stem = word[:-2]
        candidates = [stem + "e", stem]
        if len(stem) >= 3 and stem[-1] == stem[-2]:
            candidates.append(stem[:-1])
        for cand in candidates:
            if cand in self._known_verbs:
                return cand
        if stem and stem[-1] in "vcgzu":
            return stem + "e"
        if len(stem) >= 3 and stem[-1] == stem[-2] and stem[-1] in "bgmnprt":
            return stem[:-1]
        return stem



    def _standardize_verbs(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        first = words[0].strip(".,;:!?")
        base = self._to_base_form(first)
        canonical = self.VERB_NORMALIZATION.get(base) if base else None
        if canonical:
            words[0] = canonical
        return " ".join(words)

    def _to_base_form(self, word: str) -> Optional[str]:

        if word in self._known_verbs:
            return word
        if word in self.IRREGULAR_VERB_FORMS:
            return self.IRREGULAR_VERB_FORMS[word]
        candidates: List[str] = []
        if word.endswith("ies"):
            candidates.append(word[:-3] + "y")
        if word.endswith("es"):
            candidates.append(word[:-2])
        if word.endswith("s"):
            candidates.append(word[:-1])
        if word.endswith("ing"):
            candidates.extend([word[:-3], word[:-3] + "e"])
        if word.endswith("ed"):
            candidates.extend([word[:-2], word[:-1]])
        for cand in candidates:
            if cand in self._known_verbs:
                return cand
        return None



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

    def match_categories(self, text: str) -> List[str]:

        text_tokens = {self._stem(t) for t in re.findall(r"[a-zA-Z]+", text.lower())}
        matched: List[Tuple[str, bool]] = []

        for cat, subtokens in self._category_stem_tokens.items():
            if not subtokens:
                continue
            hits = sum(1 for s in subtokens if s in text_tokens)
            is_full = hits == len(subtokens)
            evidence = self._category_evidence_stems.get(cat, set())

            if len(subtokens) == 1:
                if not is_full:
                    continue
                if self.require_evidence_for_single_word and not (evidence & text_tokens):
                    continue
                matched.append((cat, True))
            elif is_full:
                matched.append((cat, True))
            elif hits / len(subtokens) >= self._PARTIAL_MATCH_MIN_RATIO and (evidence & text_tokens):
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

    def _inject_api_keywords(self, text: str) -> Tuple[str, List[str]]:

        if not self.DOMAIN_KEYWORDS:
            return text, []

        selected: List[str] = []
        seen_heads: Set[str] = set()
        for cat in self.match_categories(text):
            phrase = self.DOMAIN_KEYWORDS[cat]
            head = phrase.split()[0].lower() if phrase.split() else ""
            if head in seen_heads:
                continue
            seen_heads.add(head)
            selected.append(phrase)
            if len(selected) >= self.MAX_INJECTED_PHRASES:
                break

        if not selected:
            return text, []
        return text.rstrip(". ") + " (" + " | ".join(selected) + ").", selected




if __name__ == "__main__":
    demo_tasks = [
        {"task_id": "task_1", "name": "Clarify shipping",
         "description": "The secretary clarifies who will do the shipping"},
        {"task_id": "task_2", "name": "Write package label",
         "description": "A package label is written by the secretary"},
        {"task_id": "task_3", "name": "Package goods",
         "description": "Goods are packaged by warehouse staff"},
        {"task_id": "task_4", "name": "Take insurance",
         "description": "The logistics department head takes insurance for the parcel"},
        {"task_id": "task_5", "name": "Invite bids",
         "description": "The manager invites bids from carriers"},
    ]

    enhancer = TaskEnhancer()
    enhanced = enhancer.enhance_batch(demo_tasks)

    for task in enhanced:
        print("\n[%s]" % task["task_id"])
        print("  original: %s" % task["original_description"])
        print("  enhanced: %s" % task["description"])

    summary = enhancer.get_enhancement_summary(enhanced)
    print("\nSummary:")
    for key, value in summary.items():
        print("  %s: %s" % (key, value))
