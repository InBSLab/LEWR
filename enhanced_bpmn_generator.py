import os
import json
import re
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


ENHANCED_BPMN_PROMPT = '''# BPMN JSON Representation Guide (Enhanced v2.3)

The BPMN JSON uses a hierarchical structure where gateways contain their branches directly.

## Core Elements

### 1. Tasks
- **userTask**: Human decisions/reviews
- **serviceTask**: Automated system operations
- **sendTask**: Sending communications
- **receiveTask**: Waiting for input
- **manualTask**: Physical work without IT
- **businessRuleTask**: Rule engine decisions
- **scriptTask**: Script execution

```json
{"type": "userTask", "id": "task1", "label": "Review application"}
```

### 2. Events
```json
{"type": "startEvent", "id": "start"}
{"type": "endEvent", "id": "end"}
```

### 3. Gateways

#### Exclusive Gateway (XOR) — "if", "depending on", "either/or"
```json
{
    "type": "exclusiveGateway", "id": "exclusive1",
    "label": "Decision?", "has_join": true,
    "branches": [
        {"condition": "Condition A", "path": [{"type": "userTask", "id": "t1", "label": "Task A"}]},
        {"condition": "Condition B", "path": [{"type": "userTask", "id": "t2", "label": "Task B"}]}
    ]
}
```

#### Parallel Gateway (AND) — "meanwhile", "simultaneously", "in parallel"
```json
{
    "type": "parallelGateway", "id": "parallel1",
    "branches": [
        [{"type": "manualTask", "id": "t3", "label": "Branch task 1"}],
        [{"type": "userTask", "id": "t4", "label": "Branch task 2"}]
    ]
}
```

#### Inclusive Gateway (OR) — "if necessary", "optionally", "may"
```json
{
    "type": "inclusiveGateway", "id": "inclusive1",
    "label": "Optional?", "has_join": true,
    "branches": [
        {"condition": "If needed", "path": [{"type": "userTask", "id": "t5", "label": "Optional task"}]},
        {"condition": "Default", "path": [], "is_default": true}
    ]
}
```

## Complete Process Structure
```json
{
    "process": [
        {"type": "startEvent", "id": "start"},
        {"type": "userTask", "id": "task1", "label": "First task"},
        {"type": "exclusiveGateway", "id": "gw1", "label": "Decision?", "has_join": true, "branches": [...]},
        {"type": "endEvent", "id": "end"}
    ]
}
```

## ═══════════════════════════════════════
## STRUCTURAL & MODELING RULES
## ═══════════════════════════════════════

### RULE A — Independent Item/Category Selection
When the text describes choosing among or handling several independent
item categories (e.g. product types, service tiers, request types) that
are each processed through their own steps, use a parallelGateway with
ONE BRANCH PER category that the text actually mentions. The number of
branches and the number of tasks inside each branch must come only from
what the text states or clearly implies for that category — never pad
to reach a fixed branch count or task count.

### RULE B — Multi-Role Parallel
Use parallelGateway with one branch per role/department.

### RULE C — Nested Conditions
Nest gateways inside branch paths.

### RULE D — Task Granularity
One task = one atomic action. Never collapse steps.

### RULE E — Structural Validity (CRITICAL)
- EVERY exclusiveGateway MUST have "has_join": true
- EVERY inclusiveGateway MUST have "has_join": true
- EVERY gateway MUST have at least 2 branches
- ALL element IDs must be globally unique
'''



ENHANCED_EXAMPLES = '''# Complete BPMN Examples (v2.3)

## Example 1: Illustrative example — exclusive branch containing a sub-sequence, plus a parallel branch with a nested optional (inclusive) step
**This shows the JSON schema/format only. Do not copy this content for other inputs — always derive tasks and structure from the actual text being processed.**

```json
{
    "process": [
        {"type": "startEvent", "id": "start"},
        {"type": "userTask", "id": "task_clarify", "label": "Clarify ticket category"},
        {
            "type": "exclusiveGateway", "id": "eg_category",
            "label": "Hardware or software issue?", "has_join": true,
            "branches": [
                {
                    "condition": "Hardware issue",
                    "path": [
                        {"type": "userTask", "id": "task_assign_tech", "label": "Assign field technician"},
                        {"type": "serviceTask", "id": "task_order_part", "label": "Order replacement part"},
                        {"type": "manualTask", "id": "task_install_part", "label": "Install replacement part"}
                    ]
                },
                {
                    "condition": "Software issue",
                    "path": [{"type": "scriptTask", "id": "task_patch", "label": "Deploy remote software patch"}]
                }
            ]
        },
        {
            "type": "parallelGateway", "id": "pg_wrapup",
            "branches": [
                [
                    {"type": "serviceTask", "id": "task_log", "label": "Update ticket status log"},
                    {
                        "type": "inclusiveGateway", "id": "ig_callback",
                        "label": "Callback requested?", "has_join": true,
                        "branches": [
                            {"condition": "Callback requested", "path": [{"type": "userTask", "id": "task_callback", "label": "Schedule callback call"}]},
                            {"condition": "No callback", "path": [], "is_default": true}
                        ]
                    }
                ],
                [{"type": "sendTask", "id": "task_notify", "label": "Notify requester ticket is resolved"}]
            ]
        },
        {"type": "userTask", "id": "task_close", "label": "Close ticket"},
        {"type": "endEvent", "id": "end"}
    ]
}
```

## Example 2: Illustrative example — single exclusive gateway with a short branch on each side
**This shows the JSON schema/format only. Do not copy this content for other inputs — always derive tasks and structure from the actual text being processed.**

```json
{
    "process": [
        {"type": "startEvent", "id": "start"},
        {"type": "receiveTask", "id": "task_receive", "label": "Receive bug report"},
        {"type": "userTask", "id": "task_register", "label": "Register report in tracker"},
        {"type": "businessRuleTask", "id": "task_triage", "label": "Assess severity level"},
        {
            "type": "exclusiveGateway", "id": "eg_severity",
            "label": "Severity critical?", "has_join": true,
            "branches": [
                {
                    "condition": "Critical severity",
                    "path": [
                        {"type": "userTask", "id": "task_escalate", "label": "Escalate to on-call engineer"},
                        {"type": "serviceTask", "id": "task_hotfix", "label": "Deploy hotfix"},
                        {"type": "sendTask", "id": "task_notify_fixed", "label": "Notify reporter of fix"}
                    ]
                },
                {
                    "condition": "Low severity",
                    "path": [
                        {"type": "userTask", "id": "task_backlog", "label": "Add report to backlog"}
                    ]
                }
            ]
        },
        {"type": "userTask", "id": "task_close", "label": "Close and archive report"},
        {"type": "endEvent", "id": "end"}
    ]
}
```

## Example 3: Illustrative example — exclusive gateway containing a nested parallel gateway
**This shows the JSON schema/format only. Do not copy this content for other inputs — always derive tasks and structure from the actual text being processed.**

```json
{
    "process": [
        {"type": "startEvent", "id": "start"},
        {"type": "receiveTask", "id": "task_receive", "label": "Receive paper submission"},
        {"type": "businessRuleTask", "id": "task_screen", "label": "Run automated formatting check"},
        {"type": "userTask", "id": "task_chair", "label": "Program chair performs initial screening"},
        {
            "type": "exclusiveGateway", "id": "eg_score",
            "label": "Initial score above threshold?", "has_join": true,
            "branches": [
                {
                    "condition": "Above threshold",
                    "path": [
                        {
                            "type": "parallelGateway", "id": "pg_reviews",
                            "branches": [
                                [
                                    {"type": "userTask", "id": "task_review1", "label": "First reviewer evaluates paper"},
                                    {"type": "userTask", "id": "task_review2", "label": "Second reviewer evaluates paper"},
                                    {"type": "userTask", "id": "task_review3", "label": "Third reviewer evaluates paper"}
                                ],
                                [
                                    {"type": "serviceTask", "id": "task_plagiarism", "label": "Run plagiarism check"},
                                    {"type": "serviceTask", "id": "task_similarity", "label": "Compute topic similarity score"},
                                    {"type": "userTask", "id": "task_area_match", "label": "Confirm track/area match"}
                                ]
                            ]
                        },
                        {"type": "userTask", "id": "task_discuss", "label": "Committee holds discussion"}
                    ]
                },
                {"condition": "Below threshold", "path": []}
            ]
        },
        {
            "type": "exclusiveGateway", "id": "eg_decision",
            "label": "Final decision?", "has_join": true,
            "branches": [
                {"condition": "Accepted", "path": [{"type": "sendTask", "id": "task_accept", "label": "Send acceptance notification"}]},
                {"condition": "Rejected", "path": [{"type": "sendTask", "id": "task_reject", "label": "Send rejection notification"}]}
            ]
        },
        {"type": "endEvent", "id": "end"}
    ]
}
```

## Example 4: Illustrative example — parallel gateway with several branches, one per role/department
**This shows the JSON schema/format only. Do not copy this content for other inputs — always derive tasks and structure from the actual text being processed.**

```json
{
    "process": [
        {"type": "startEvent", "id": "start"},
        {"type": "userTask", "id": "task_welcome", "label": "HR sends welcome email"},
        {"type": "userTask", "id": "task_paperwork", "label": "New hire completes onboarding paperwork"},
        {
            "type": "parallelGateway", "id": "pg_departments",
            "branches": [
                [
                    {"type": "serviceTask", "id": "t_it_account", "label": "Create IT account"},
                    {"type": "manualTask", "id": "t_it_hardware", "label": "Prepare laptop and hardware"}
                ],
                [
                    {"type": "manualTask", "id": "t_fac_desk", "label": "Assign desk space"},
                    {"type": "manualTask", "id": "t_fac_badge", "label": "Issue building access badge"}
                ],
                [
                    {"type": "userTask", "id": "t_pay_setup", "label": "Set up payroll direct deposit"},
                    {"type": "userTask", "id": "t_pay_benefits", "label": "Enroll in benefits plan"}
                ]
            ]
        },
        {"type": "userTask", "id": "task_orientation", "label": "Attend orientation session"},
        {"type": "userTask", "id": "task_checkin", "label": "Manager holds first-week check-in"},
        {"type": "endEvent", "id": "end"}
    ]
}
```
'''



GENERIC_STRUCTURE_HINT = """
STRUCTURE GUIDANCE:
- Model every conditional described in the text ("if", "in case of",
  "depending on", etc.) as an exclusiveGateway or inclusiveGateway with
  has_join: true, using the branches actually described in the text.
- Model activities the text describes as happening concurrently or in any
  order (e.g. "meanwhile", "at the same time", "in parallel", "in any order")
  as a parallelGateway with one branch per concurrent activity/role.
- Do not invent tasks, roles, or conditions that are not stated or clearly
  implied by the text, and do not omit steps that are explicitly stated.
"""



@dataclass
class BPMNValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]



class EnhancedBPMNGenerator:


    def __init__(self, api_key: str = None, base_url: str = None, debug: bool = True):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or "https://api.deepseek.com/v1"
        self.debug = debug
        self.system_prompt = self._build_enhanced_prompt()
        self.stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_gateways_detected": 0,
            "total_llm_attempts": 0,
            "requests_needing_retry": 0
        }

    def _build_enhanced_prompt(self) -> str:
        return f"""You are a world-class BPMN expert.
Convert natural language process descriptions into structured BPMN JSON.

{ENHANCED_BPMN_PROMPT}

{ENHANCED_EXAMPLES}

---

## CRITICAL REQUIREMENTS (summary)

1. **RULE A — Independent categories:** parallelGateway with one branch per
   category the text actually describes; the branch count and the task
   count inside each branch come only from the text, never from a fixed
   target.
2. **RULE B — Multi-role:** One branch per department/role.
3. **RULE C — Nested logic:** Nest gateways inside branches.
4. **RULE D — Granularity:** One task = one atomic action.
5. **RULE E — Structural validity:**
   - EVERY exclusiveGateway and inclusiveGateway MUST have "has_join": true
   - EVERY gateway MUST have at least 2 branches
   - ALL element IDs must be globally unique

6. **TASK COUNT DISCIPLINE:** There is NO fixed or target task count for
   any scenario, topic, or domain. The number of tasks, branches, and
   gateways must be derived only from what this specific input text
   states or clearly implies. Never let the subject matter or topic of
   the process push the count toward a remembered number — count only
   what this specific text describes.

7. **Output format:** Output ONLY valid JSON, no prose outside the code block.

Now analyze the input and generate BPMN JSON with MAXIMUM accuracy."""

    def generate(
        self,
        process_description: str,
        max_retries: int = 3,
        validate: bool = True
    ) -> Dict[str, Any]:
        self.stats["total_requests"] += 1

        if self.debug:
            print("\n" + "="*70)
            print("Enhanced BPMN Generator v2.3-fix - Starting Generation")
            print("="*70)

        raw_bpmn = self._generate_with_llm(process_description, max_retries)

        if not raw_bpmn or "process" not in raw_bpmn:
            self.stats["failed_generations"] += 1
            return self._generate_fallback(process_description)


        raw_bpmn = self._post_process_bpmn(raw_bpmn, process_description)

        self.stats["successful_generations"] += 1

        if validate:
            validation = self._validate_bpmn(raw_bpmn)
        else:
            validation = BPMNValidationResult(True, [], [], {})

        enhanced_info = self._extract_enhanced_info(raw_bpmn)
        evaluation_format = self._convert_to_evaluation_format(raw_bpmn, enhanced_info)
        self.stats["total_gateways_detected"] += enhanced_info["gateway_count"]

        if self.debug:
            print(f"\n[Statistics]")
            print(f"  Tasks: {enhanced_info['task_count']}")
            print(f"  Gateways: {enhanced_info['gateway_count']}")
            print(f"    - Exclusive: {enhanced_info['exclusive_count']}")
            print(f"    - Parallel: {enhanced_info['parallel_count']}")
            print(f"    - Inclusive: {enhanced_info['inclusive_count']}")

        return {
            "bpmn_workflow": raw_bpmn,
            "evaluation_format": evaluation_format,
            "enhanced_info": enhanced_info,
            "validation": {
                "is_valid": validation.is_valid,
                "errors": validation.errors,
                "warnings": validation.warnings
            },
            "metadata": {
                "generator": "EnhancedBPMNGenerator v2.3-fix",
                "timestamp": self._get_timestamp(),
                "input_length": len(process_description)
            }
        }

    def _generate_with_llm(self, process_description: str, max_retries: int) -> Optional[Dict]:

        user_prompt = f"""Process description:
{process_description}
{GENERIC_STRUCTURE_HINT}

Generate the complete BPMN JSON following ALL rules above.
Remember:
- RULE E: every exclusiveGateway and inclusiveGateway MUST have "has_join": true
- Base the task count and structure purely on what this specific text
  describes; do not target any fixed number of tasks.

BPMN JSON:"""

        for attempt in range(max_retries):
            self.stats["total_llm_attempts"] += 1
            try:
                if self.debug:
                    print(f"\n[LLM Call] Attempt {attempt + 1}/{max_retries}")

                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        "max_tokens": 2500,
                        "temperature": 0.3
                    },
                    timeout=90
                )

                if response.status_code == 200:
                    result = response.json()
                    content = result["choices"][0]["message"]["content"]
                    bpmn_workflow = self._parse_json(content)
                    if bpmn_workflow and "process" in bpmn_workflow:
                        if self.debug:
                            print(f"  ✓ Successfully generated BPMN")
                        if attempt > 0:
                            self.stats["requests_needing_retry"] += 1
                        return bpmn_workflow
                    else:
                        raise ValueError("Invalid BPMN structure")
                else:
                    raise Exception(f"API error: {response.status_code}")

            except Exception as e:
                if self.debug:
                    print(f"  ✗ Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return None

        return None

    def _post_process_bpmn(self, bpmn: Dict, description: str) -> Dict:

        process = bpmn.get("process", [])
        self._fix_has_join_recursive(process)
        self._ensure_unique_ids(process)
        return bpmn

    def _fix_has_join_recursive(self, elements: List):

        for elem in elements:
            if not isinstance(elem, dict):
                continue

            elem_type = elem.get("type", "")


            if elem_type in ["exclusiveGateway", "inclusiveGateway"]:
                elem["has_join"] = True

            if "branches" in elem:
                for branch in elem["branches"]:
                    if isinstance(branch, list):
                        self._fix_has_join_recursive(branch)
                    elif isinstance(branch, dict) and "path" in branch:
                        self._fix_has_join_recursive(branch["path"])

    def _ensure_unique_ids(self, elements: List, seen_ids: set = None):
        """确保所有ID唯一"""
        if seen_ids is None:
            seen_ids = set()

        counter = [0]

        def fix_ids(elems):
            for elem in elems:
                if not isinstance(elem, dict):
                    continue
                elem_id = elem.get("id", "")
                if elem_id:
                    if elem_id in seen_ids:
                        counter[0] += 1
                        new_id = f"{elem_id}_dup{counter[0]}"
                        elem["id"] = new_id
                        seen_ids.add(new_id)
                    else:
                        seen_ids.add(elem_id)

                if "branches" in elem:
                    for branch in elem["branches"]:
                        if isinstance(branch, list):
                            fix_ids(branch)
                        elif isinstance(branch, dict) and "path" in branch:
                            fix_ids(branch["path"])

        fix_ids(elements)

    def _parse_json(self, content: str) -> Optional[Dict]:
        """解析LLM输出的JSON"""
        json_pattern = r'```(?:json)?\s*(.*?)\s*```'
        matches = re.findall(json_pattern, content, re.DOTALL)
        if matches:
            try:
                return json.loads(matches[0])
            except json.JSONDecodeError:
                pass
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        except json.JSONDecodeError:
            pass
        return None

    def _validate_bpmn(self, bpmn: Dict) -> BPMNValidationResult:
        """验证BPMN工作流"""
        errors = []
        warnings = []

        process = bpmn.get("process", [])
        if not process:
            errors.append("Empty process")
            return BPMNValidationResult(False, errors, warnings, {})

        has_start = any(e.get("type") == "startEvent" for e in process)
        has_end = self._has_end_event_recursive(process)

        if not has_start:
            errors.append("Missing startEvent")
        if not has_end:
            errors.append("Missing endEvent")

        all_ids = self._collect_all_ids(process)
        if len(all_ids) != len(set(all_ids)):
            warnings.append("Duplicate IDs found (auto-fixed)")

        stats = self._count_elements(process)

        missing_join = self._check_has_join(process)
        if missing_join:
            warnings.append(f"Gateways fixed has_join: {missing_join[:3]}")

        return BPMNValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            statistics=stats
        )

    def _check_has_join(self, elements: List) -> List[str]:
        missing = []
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            elem_type = elem.get("type", "")
            if elem_type in ["exclusiveGateway", "inclusiveGateway"]:
                if not elem.get("has_join", False):
                    missing.append(elem.get("id", "unknown"))
            if "branches" in elem:
                for branch in elem["branches"]:
                    if isinstance(branch, list):
                        missing.extend(self._check_has_join(branch))
                    elif isinstance(branch, dict) and "path" in branch:
                        missing.extend(self._check_has_join(branch["path"]))
        return missing

    def _has_end_event_recursive(self, elements: List) -> bool:
        for elem in elements:
            if elem.get("type") == "endEvent":
                return True
            if "branches" in elem:
                for branch in elem["branches"]:
                    if isinstance(branch, list):
                        if self._has_end_event_recursive(branch):
                            return True
                    elif isinstance(branch, dict) and "path" in branch:
                        if self._has_end_event_recursive(branch["path"]):
                            return True
        return False

    def _collect_all_ids(self, elements: List, ids: Optional[List] = None) -> List[str]:
        if ids is None:
            ids = []
        for elem in elements:
            if "id" in elem:
                ids.append(elem["id"])
            if "branches" in elem:
                for branch in elem["branches"]:
                    if isinstance(branch, list):
                        self._collect_all_ids(branch, ids)
                    elif isinstance(branch, dict) and "path" in branch:
                        self._collect_all_ids(branch["path"], ids)
        return ids

    def _count_elements(self, elements: List) -> Dict[str, int]:
        stats = {"total_elements": 0, "tasks": 0, "gateways": 0, "events": 0}
        task_types = {"task", "userTask", "serviceTask", "sendTask",
                      "receiveTask", "businessRuleTask", "manualTask", "scriptTask"}
        gateway_types = {"exclusiveGateway", "parallelGateway",
                         "inclusiveGateway", "eventBasedGateway"}
        event_types = {"startEvent", "endEvent",
                       "intermediateCatchEvent", "intermediateThrowEvent"}

        def count_recursive(elems):
            for elem in elems:
                stats["total_elements"] += 1
                t = elem.get("type", "")
                if t in task_types:
                    stats["tasks"] += 1
                elif t in gateway_types:
                    stats["gateways"] += 1
                elif t in event_types:
                    stats["events"] += 1
                if "branches" in elem:
                    for branch in elem["branches"]:
                        if isinstance(branch, list):
                            count_recursive(branch)
                        elif isinstance(branch, dict) and "path" in branch:
                            count_recursive(branch["path"])

        count_recursive(elements)
        return stats

    def _extract_enhanced_info(self, bpmn: Dict) -> Dict[str, Any]:
        info = {
            "task_count": 0, "gateway_count": 0,
            "exclusive_count": 0, "parallel_count": 0,
            "inclusive_count": 0, "eventbased_count": 0,
            "tasks": [], "gateways": []
        }

        def extract_recursive(elements, depth=0):
            for elem in elements:
                elem_type = elem.get("type", "")

                if elem_type in {"task", "userTask", "serviceTask", "sendTask",
                                 "receiveTask", "businessRuleTask", "manualTask", "scriptTask"}:
                    info["tasks"].append({
                        "id": elem.get("id"),
                        "name": elem.get("label", ""),
                        "type": elem_type,
                        "depth": depth
                    })
                    info["task_count"] += 1

                elif elem_type in {"exclusiveGateway", "parallelGateway",
                                   "inclusiveGateway", "eventBasedGateway"}:
                    gateway_info = {
                        "id": elem.get("id"),
                        "type": elem_type.replace("Gateway", ""),
                        "label": elem.get("label", ""),
                        "has_join": elem.get("has_join", False),
                        "branch_count": len(elem.get("branches", [])),
                        "depth": depth,
                        "conditions": []
                    }
                    if "branches" in elem:
                        for branch in elem["branches"]:
                            if isinstance(branch, dict) and "condition" in branch:
                                gateway_info["conditions"].append(branch["condition"])
                    info["gateways"].append(gateway_info)
                    info["gateway_count"] += 1
                    if elem_type == "exclusiveGateway":
                        info["exclusive_count"] += 1
                    elif elem_type == "parallelGateway":
                        info["parallel_count"] += 1
                    elif elem_type == "inclusiveGateway":
                        info["inclusive_count"] += 1
                    elif elem_type == "eventBasedGateway":
                        info["eventbased_count"] += 1

                if "branches" in elem:
                    for branch in elem["branches"]:
                        if isinstance(branch, list):
                            extract_recursive(branch, depth + 1)
                        elif isinstance(branch, dict) and "path" in branch:
                            extract_recursive(branch["path"], depth + 1)

        extract_recursive(bpmn.get("process", []))
        return info

    def _convert_to_evaluation_format(self, bpmn: Dict, enhanced_info: Dict) -> Dict[str, Any]:

        tasks = enhanced_info["tasks"]
        dependencies = self._build_complete_dependencies(bpmn.get("process", []))
        gateway_contexts = self._build_gateway_contexts(enhanced_info["gateways"], dependencies)

        gateways = []
        for gw in enhanced_info["gateways"]:
            gw_type = gw["type"]


            original_id = gw["id"]  # 例如 "eg_amount", "pg_parallel", "ig_insurance"

            gw_tasks = self._extract_gateway_tasks(bpmn, original_id)
            entry_point = self._infer_gateway_entry(bpmn, original_id)
            merge_point = self._infer_gateway_merge(bpmn, original_id) if gw["has_join"] else None

            gateway_obj = {
                "id": original_id,
                "type": gw_type,
                "tasks": {
                    "tasks": gw_tasks,
                    "type": gw_type,
                    "pattern": "fork",
                    "id": original_id,
                    "entry_point": entry_point
                }
            }
            if merge_point:
                gateway_obj["tasks"]["merge_point"] = merge_point
            if gw["conditions"]:
                gateway_obj["tasks"]["conditions"] = gw["conditions"]
            gateways.append(gateway_obj)

        return {
            "tasks": [
                {"task_id": t["id"], "name": t["name"], "type": t["type"], "description": t["name"]}
                for t in tasks
            ],
            "dependencies": dependencies,
            "gateways": gateways,
            "contexts": gateway_contexts
        }

    def _build_complete_dependencies(self, process: List) -> List[Dict]:
        dependencies = []

        def build_recursive(elements, predecessor_id=None):
            if not elements:
                return predecessor_id
            prev_id = predecessor_id

            for elem in elements:
                if not isinstance(elem, dict):
                    continue
                elem_type = elem.get("type", "")
                elem_id = elem.get("id")
                if not elem_id:
                    continue

                if prev_id:
                    dependencies.append({"source": prev_id, "target": elem_id})

                if elem_type in ["exclusiveGateway", "parallelGateway",
                                  "inclusiveGateway", "eventBasedGateway"]:
                    gateway_id = elem_id
                    branches = elem.get("branches", [])
                    branch_endings = []

                    for branch in branches:
                        if isinstance(branch, list):
                            last_id = build_recursive(branch, gateway_id)
                            if last_id:
                                branch_endings.append(last_id)
                        elif isinstance(branch, dict) and "path" in branch:
                            last_id = build_recursive(branch["path"], gateway_id)
                            if last_id:
                                branch_endings.append(last_id)

                    if elem.get("has_join", False) and branch_endings:
                        join_id = f"{gateway_id}_join"
                        for ending_id in branch_endings:
                            dependencies.append({"source": ending_id, "target": join_id})
                        prev_id = join_id
                    elif branch_endings:
                        prev_id = branch_endings[-1]
                    else:
                        prev_id = gateway_id
                else:
                    prev_id = elem_id

            return prev_id

        build_recursive(process)
        return dependencies

    def _build_gateway_contexts(self, gateways: List[Dict], dependencies: List[Dict]) -> List[Dict]:
        contexts = []
        in_degree = {}
        out_degree = {}
        predecessors = {}
        successors = {}

        for dep in dependencies:
            source = dep.get("source", "")
            target = dep.get("target", "")
            if not source or not target:
                continue
            out_degree[source] = out_degree.get(source, 0) + 1
            successors.setdefault(source, []).append(target)
            in_degree[target] = in_degree.get(target, 0) + 1
            predecessors.setdefault(target, []).append(source)

        for gw in gateways:
            gw_id = gw.get("id", "")
            gw_type = gw.get("type", "unknown")
            join_id = f"{gw_id}_join"

            contexts.append({
                "id": gw_id, "type": gw_type,
                "in_degree": in_degree.get(gw_id, 0),
                "out_degree": out_degree.get(gw_id, 0),
                "predecessors": predecessors.get(gw_id, []),
                "successors": successors.get(gw_id, [])
            })

            if join_id in in_degree or join_id in out_degree:
                contexts.append({
                    "id": join_id, "type": f"{gw_type}_join",
                    "in_degree": in_degree.get(join_id, 0),
                    "out_degree": out_degree.get(join_id, 0),
                    "predecessors": predecessors.get(join_id, []),
                    "successors": successors.get(join_id, [])
                })

        return contexts

    def _extract_gateway_tasks(self, bpmn: Dict, gateway_id: str) -> List[str]:
        tasks = []

        def find_gateway_and_extract(elements):
            for elem in elements:
                if elem.get("id") == gateway_id and "branches" in elem:
                    for branch in elem["branches"]:
                        if isinstance(branch, list):
                            tasks.extend(self._extract_task_ids(branch))
                        elif isinstance(branch, dict) and "path" in branch:
                            tasks.extend(self._extract_task_ids(branch["path"]))
                    return True
                if "branches" in elem:
                    for branch in elem["branches"]:
                        if isinstance(branch, list):
                            if find_gateway_and_extract(branch): return True
                        elif isinstance(branch, dict) and "path" in branch:
                            if find_gateway_and_extract(branch["path"]): return True
            return False

        find_gateway_and_extract(bpmn.get("process", []))
        return tasks

    def _extract_task_ids(self, elements: List) -> List[str]:
        task_ids = []
        task_types = {"task", "userTask", "serviceTask", "sendTask",
                      "receiveTask", "businessRuleTask", "manualTask", "scriptTask"}
        for elem in elements:
            if elem.get("type") in task_types:
                task_ids.append(elem.get("id"))
            if "branches" in elem:
                for branch in elem["branches"]:
                    if isinstance(branch, list):
                        task_ids.extend(self._extract_task_ids(branch))
                    elif isinstance(branch, dict) and "path" in branch:
                        task_ids.extend(self._extract_task_ids(branch["path"]))
        return task_ids

    def _infer_gateway_entry(self, bpmn: Dict, gateway_id: str) -> Optional[str]:
        process = bpmn.get("process", [])
        for i, elem in enumerate(process):
            if elem.get("id") == gateway_id:
                if i > 0:
                    return process[i - 1].get("id")
                break
        return None

    def _infer_gateway_merge(self, bpmn: Dict, gateway_id: str) -> Optional[str]:
        process = bpmn.get("process", [])
        for i, elem in enumerate(process):
            if elem.get("id") == gateway_id:
                if i < len(process) - 1:
                    return process[i + 1].get("id")
                break
        return None

    def _generate_fallback(self, process_description: str) -> Dict[str, Any]:
        if self.debug:
            print("\n[Fallback] Generating simple template")
        fallback_bpmn = {
            "process": [
                {"type": "startEvent", "id": "start"},
                {"type": "userTask", "id": "task1", "label": "Process request"},
                {"type": "serviceTask", "id": "task2", "label": "Execute action"},
                {"type": "userTask", "id": "task3", "label": "Complete process"},
                {"type": "endEvent", "id": "end"}
            ]
        }
        return {
            "bpmn_workflow": fallback_bpmn,
            "evaluation_format": {
                "tasks": [
                    {"task_id": "task1", "name": "Process request", "type": "userTask", "description": "Process request"},
                    {"task_id": "task2", "name": "Execute action", "type": "serviceTask", "description": "Execute action"},
                    {"task_id": "task3", "name": "Complete process", "type": "userTask", "description": "Complete process"}
                ],
                "dependencies": [
                    {"source": "task1", "target": "task2"},
                    {"source": "task2", "target": "task3"}
                ],
                "gateways": []
            },
            "enhanced_info": {
                "task_count": 3, "gateway_count": 0,
                "exclusive_count": 0, "parallel_count": 0,
                "inclusive_count": 0, "eventbased_count": 0,
                "tasks": [], "gateways": []
            },
            "validation": {"is_valid": True, "errors": [], "warnings": ["Using fallback template"]},
            "metadata": {
                "generator": "EnhancedBPMNGenerator v2.3-fix (Fallback)",
                "timestamp": self._get_timestamp(),
                "is_fallback": True
            }
        }

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()

    def get_statistics(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_generations"] / self.stats["total_requests"]
                if self.stats["total_requests"] > 0 else 0
            )
        }


def create_enhanced_generator(api_key: str = None, debug: bool = True) -> EnhancedBPMNGenerator:
    return EnhancedBPMNGenerator(api_key=api_key, debug=debug)
