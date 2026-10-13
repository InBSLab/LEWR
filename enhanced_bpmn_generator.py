import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

import requests


TASK_TYPES = frozenset({
    "userTask", "serviceTask", "sendTask", "receiveTask",
    "manualTask", "businessRuleTask", "scriptTask",
})

GATEWAY_TYPES = frozenset({"exclusiveGateway", "parallelGateway", "inclusiveGateway"})


CONDITIONAL_GATEWAY_TYPES = frozenset({"exclusiveGateway", "inclusiveGateway"})



ENHANCED_BPMN_PROMPT = '''# BPMN JSON Representation Guide

The BPMN JSON is hierarchical: every gateway contains its branches directly,
and a branch may contain further gateways (nesting).

## Elements

### 1. Tasks
Every task has "type", "id", "label" (a short task name) and "description"
(one sentence describing the action).

Task types:
- userTask: human decisions or reviews
- serviceTask: automated system operations
- sendTask: sending communications
- receiveTask: waiting for input
- manualTask: physical work without IT support
- businessRuleTask: rule-engine decisions
- scriptTask: script execution

```json
{"type": "userTask", "id": "task_1", "label": "Review application", "description": "Review the submitted application."}
```

### 2. Events
```json
{"type": "startEvent", "id": "start"}
{"type": "endEvent", "id": "end"}
```

### 3. Gateways
Detect gateways from the lexical triggers below. A gateway is created only
when the description contains a trigger or an unambiguous equivalent.

#### Exclusive gateway (XOR)
Triggered by conditional expressions such as "if", "depending on",
"in case of", "when".
```json
{
    "type": "exclusiveGateway", "id": "exclusive_1",
    "label": "Decision?", "has_join": true,
    "branches": [
        {"condition": "Condition A", "path": [{"type": "userTask", "id": "task_2", "label": "Task A", "description": "Do A."}]},
        {"condition": "Condition B", "path": [{"type": "userTask", "id": "task_3", "label": "Task B", "description": "Do B."}]}
    ]
}
```

#### Parallel gateway (AND)
Triggered by concurrency indicators such as "meanwhile", "at the same time",
"in parallel". Branches are plain task sequences without conditions.
```json
{
    "type": "parallelGateway", "id": "parallel_1",
    "label": "Concurrent activities", "has_join": true,
    "branches": [
        [{"type": "manualTask", "id": "task_4", "label": "Branch task 1", "description": "Do the first activity."}],
        [{"type": "userTask", "id": "task_5", "label": "Branch task 2", "description": "Do the second activity."}]
    ]
}
```

#### Inclusive gateway (OR)
Triggered by optional constructs such as "if necessary", "may", "optionally".
```json
{
    "type": "inclusiveGateway", "id": "inclusive_1",
    "label": "Optional?", "has_join": true,
    "branches": [
        {"condition": "If necessary", "path": [{"type": "userTask", "id": "task_6", "label": "Optional task", "description": "Do the optional step."}]},
        {"condition": "Otherwise", "path": []}
    ]
}
```

## Complete Process Structure
```json
{
    "process": [
        {"type": "startEvent", "id": "start"},
        {"type": "userTask", "id": "task_1", "label": "First task", "description": "Do the first step."},
        {"type": "exclusiveGateway", "id": "exclusive_1", "label": "Decision?", "has_join": true, "branches": []},
        {"type": "endEvent", "id": "end"}
    ]
}
```

## Modeling Rules

1. **Gateway detection.** Detect exclusive, parallel and inclusive gateways
   from the lexical triggers listed above. Do not invent gateways, tasks,
   roles or conditions that the text does not state or clearly imply, and do
   not omit steps that are explicitly stated.
2. **Branch conditions.** Extract the exact branching condition of every
   exclusive and inclusive branch verbatim from the description.
3. **Identifiers.** Use the naming convention task_1, task_2, ... for tasks
   (in order of appearance) and exclusive_1, parallel_1, inclusive_1, ... for
   gateways (numbered per gateway type). Events use "start" and "end". All
   element IDs must be globally unique.
4. **Nesting.** Encode nested gateways recursively inside the branch paths of
   the enclosing gateway.
5. **Granularity.** One task is one atomic action. Never collapse several
   steps into one task.
6. **Structural validity.**
   - Exactly one startEvent and at least one endEvent.
   - Every gateway has "has_join": true and at least two branches.
   - Every branch of an exclusive or inclusive gateway has a non-empty
     "condition", so that the number of conditions equals the number of
     branches.
'''


def _build_examples() -> str:


    start = {"type": "startEvent", "id": "start"}
    end = {"type": "endEvent", "id": "end"}

    def scope():
        counts = {"task": 0, "exclusive": 0, "parallel": 0, "inclusive": 0}

        def task(kind: str, label: str, description: str) -> Dict[str, Any]:
            counts["task"] += 1
            return {"type": kind, "id": "task_%d" % counts["task"],
                    "label": label, "description": description}

        def gateway(kind: str, label: str, branches: List[Any]) -> Dict[str, Any]:
            counts[kind] += 1
            return {"type": kind + "Gateway", "id": "%s_%d" % (kind, counts[kind]),
                    "label": label, "has_join": True, "branches": branches}

        def branch(condition: str, *elements: Dict[str, Any]) -> Dict[str, Any]:
            return {"condition": condition, "path": list(elements)}

        return task, gateway, branch


    task, gateway, branch = scope()
    example_1 = {"process": [
        start,
        task("userTask", "Clarify ticket category",
             "Determine whether the ticket concerns a hardware or a software issue."),
        gateway("exclusive", "Hardware or software issue?", [
            branch("Hardware issue",
                   task("userTask", "Assign field technician", "Assign a field technician to the ticket."),
                   task("serviceTask", "Order replacement part", "Order the replacement part from the supplier."),
                   task("manualTask", "Install replacement part", "Install the replacement part on site.")),
            branch("Software issue",
                   task("scriptTask", "Deploy remote software patch", "Deploy the software patch remotely.")),
        ]),
        gateway("parallel", "Wrap-up activities", [
            [task("serviceTask", "Update ticket status log", "Record the new status in the ticket log."),
             gateway("inclusive", "Callback requested?", [
                 branch("Callback requested",
                        task("userTask", "Schedule callback call", "Schedule a callback call with the requester.")),
                 branch("No callback"),
             ])],
            [task("sendTask", "Notify requester", "Notify the requester that the ticket is resolved.")],
        ]),
        task("userTask", "Close ticket", "Close the ticket."),
        end,
    ]}


    task, gateway, branch = scope()
    example_2 = {"process": [
        start,
        task("receiveTask", "Receive bug report", "Receive the bug report from the reporter."),
        task("userTask", "Register report", "Register the report in the tracker."),
        task("businessRuleTask", "Assess severity", "Assess the severity level of the report."),
        gateway("exclusive", "Severity critical?", [
            branch("Critical severity",
                   task("userTask", "Escalate report", "Escalate the report to the on-call engineer."),
                   task("serviceTask", "Deploy hotfix", "Deploy the hotfix."),
                   task("sendTask", "Notify reporter", "Notify the reporter that the bug is fixed.")),
            branch("Low severity",
                   task("userTask", "Add report to backlog", "Add the report to the backlog.")),
        ]),
        task("userTask", "Close report", "Close and archive the report."),
        end,
    ]}


    task, gateway, branch = scope()
    example_3 = {"process": [
        start,
        task("receiveTask", "Receive submission", "Receive the paper submission."),
        task("businessRuleTask", "Check formatting", "Run the automated formatting check."),
        task("userTask", "Screen submission", "The program chair performs the initial screening."),
        gateway("exclusive", "Initial score above threshold?", [
            branch("Above threshold",
                   gateway("parallel", "Review activities", [
                       [task("userTask", "First review", "The first reviewer evaluates the paper."),
                        task("userTask", "Second review", "The second reviewer evaluates the paper.")],
                       [task("serviceTask", "Run plagiarism check", "Run the plagiarism check on the paper."),
                        task("userTask", "Confirm track match", "Confirm that the paper matches the track.")],
                   ]),
                   task("userTask", "Hold discussion", "The committee holds a discussion.")),
            branch("Below threshold"),
        ]),
        gateway("exclusive", "Final decision?", [
            branch("Accepted", task("sendTask", "Send acceptance", "Send the acceptance notification.")),
            branch("Rejected", task("sendTask", "Send rejection", "Send the rejection notification.")),
        ]),
        end,
    ]}


    task, gateway, branch = scope()
    example_4 = {"process": [
        start,
        task("userTask", "Send welcome email", "HR sends the welcome email."),
        task("userTask", "Complete paperwork", "The new hire completes the onboarding paperwork."),
        gateway("parallel", "Department set-up", [
            [task("serviceTask", "Create IT account", "Create the IT account."),
             task("manualTask", "Prepare hardware", "Prepare the laptop and hardware.")],
            [task("manualTask", "Assign desk", "Assign desk space."),
             task("manualTask", "Issue badge", "Issue the building access badge.")],
            [task("userTask", "Set up payroll", "Set up payroll direct deposit."),
             task("userTask", "Enroll in benefits", "Enroll the employee in the benefits plan.")],
        ]),
        task("userTask", "Attend orientation", "Attend the orientation session."),
        end,
    ]}

    headers = [
        "Exclusive branch containing a sub-sequence, plus a parallel gateway with a nested inclusive gateway",
        "Single exclusive gateway with a short branch on each side",
        "Exclusive gateway containing a nested parallel gateway",
        "Parallel gateway with one branch per department",
    ]
    note = ("This illustrates the JSON schema only. Always derive tasks and "
            "structure from the description being processed.")

    parts = ["# Complete BPMN Examples", ""]
    for i, (header, example) in enumerate(zip(headers, [example_1, example_2, example_3, example_4]), 1):
        parts.append("## Example %d: %s" % (i, header))
        parts.append("**%s**" % note)
        parts.append("")
        parts.append("```json")
        parts.append(json.dumps(example, indent=2))
        parts.append("```")
        parts.append("")
    return "\n".join(parts)


ENHANCED_EXAMPLES = _build_examples()



@dataclass
class BPMNValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]



class EnhancedBPMNGenerator:
    """LLM-based BPMN generator (Stage 1) with depth-first task extraction (Stage 2)."""

    DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
    DEFAULT_MODEL = "deepseek-chat"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        debug: bool = True,
        temperature: float = 0.3,
        max_tokens: int = 2500,
        timeout: int = 90,
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.model = model or self.DEFAULT_MODEL
        self.debug = debug
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.system_prompt = self._build_system_prompt()
        self.stats = {
            "total_requests": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "total_gateways_detected": 0,
        }


    @staticmethod
    def _build_system_prompt() -> str:
        return (
            "You are an expert in BPMN process modeling.\n"
            "Convert natural language process descriptions into structured BPMN JSON.\n\n"
            + ENHANCED_BPMN_PROMPT
            + "\n"
            + ENHANCED_EXAMPLES
            + "\n---\n\n"
            "Output only valid JSON inside a single code block, with no prose "
            "outside the code block."
        )

    def generate(
        self,
        process_description: str,
        max_retries: int = 3,
        validate: bool = True,
    ) -> Dict[str, Any]:

        self.stats["total_requests"] += 1

        if self.debug:
            print("\n" + "=" * 70)
            print("Enhanced BPMN Generator - Stage 1")
            print("=" * 70)

        bpmn: Optional[Dict[str, Any]] = None
        validation = BPMNValidationResult(True, [], [], {})
        last_reason = "no attempt was made"

        for attempt in range(1, max_retries + 1):
            if self.debug:
                print("\n[LLM call] attempt %d/%d" % (attempt, max_retries))

            candidate = self._call_llm(process_description)
            if candidate is None:
                last_reason = "LLM call failed or returned unparsable output"
                continue

            self._normalize_task_types(candidate["process"])
            result = self._validate_bpmn(candidate) if validate else validation
            if result.is_valid:
                bpmn, validation = candidate, result
                break

            last_reason = "validation failed: " + "; ".join(result.errors)
            if self.debug:
                print("  validation failed: %s" % result.errors)

        if bpmn is None:
            self.stats["failed_generations"] += 1
            return self._generate_fallback(last_reason)

        self._normalize_join_flags(bpmn["process"])
        self.stats["successful_generations"] += 1
        return self._assemble_output(bpmn, validation, is_fallback=False)

    def _call_llm(self, process_description: str) -> Optional[Dict[str, Any]]:

        user_prompt = (
            "Process description:\n%s\n\n"
            "Generate the complete BPMN JSON following all modeling rules.\n\n"
            "BPMN JSON:" % process_description
        )
        try:
            response = requests.post(
                "%s/chat/completions" % self.base_url,
                headers={
                    "Authorization": "Bearer %s" % self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
                timeout=self.timeout,
            )
            if response.status_code != 200:
                raise RuntimeError("API error: %s" % response.status_code)
            content = response.json()["choices"][0]["message"]["content"]
            bpmn = self._parse_json(content)
            if not bpmn or not isinstance(bpmn.get("process"), list):
                raise ValueError("Output is not a valid BPMN structure")
            return bpmn
        except Exception as exc:
            if self.debug:
                print("  LLM attempt failed: %s" % exc)
            return None

    @staticmethod
    def _parse_json(content: str) -> Optional[Dict[str, Any]]:

        for block in re.findall(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL):
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                break
        start, end = content.find("{"), content.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass
        return None


    def _validate_bpmn(self, bpmn: Dict[str, Any]) -> BPMNValidationResult:
        """
        Structural validation:
          1) exactly one startEvent and at least one endEvent;
          2) global uniqueness of element IDs;
          3) gateway branch counts match the number of extracted conditions
             (exclusive and inclusive gateways).
        """
        errors: List[str] = []
        process = bpmn.get("process")
        if not isinstance(process, list) or not process:
            return BPMNValidationResult(False, ["Empty or missing process"], [], {})

        elements = list(self._iter_elements(process))

        n_start = sum(1 for e in elements if e.get("type") == "startEvent")
        n_end = sum(1 for e in elements if e.get("type") == "endEvent")
        if n_start != 1:
            errors.append("Expected exactly one startEvent, found %d" % n_start)
        if n_end < 1:
            errors.append("Missing endEvent")

        ids = [e.get("id") for e in elements]
        if any(not i for i in ids):
            errors.append("Element without id")
        duplicates = sorted({i for i in ids if i and ids.count(i) > 1})
        if duplicates:
            errors.append("Duplicate element IDs: %s" % duplicates[:5])

        for elem in elements:
            if elem.get("type") in CONDITIONAL_GATEWAY_TYPES:
                branches = elem.get("branches", [])
                n_conditions = sum(
                    1 for b in branches
                    if isinstance(b, dict) and str(b.get("condition", "")).strip()
                )
                if len(branches) != n_conditions:
                    errors.append(
                        "Gateway %s has %d branches but %d conditions"
                        % (elem.get("id"), len(branches), n_conditions)
                    )

        return BPMNValidationResult(
            is_valid=not errors,
            errors=errors,
            warnings=[],
            statistics=self._count_elements(elements),
        )

    @staticmethod
    def _count_elements(elements: List[Dict[str, Any]]) -> Dict[str, int]:
        stats = {"total_elements": len(elements), "tasks": 0, "gateways": 0, "events": 0}
        for elem in elements:
            etype = elem.get("type", "")
            if etype in TASK_TYPES:
                stats["tasks"] += 1
            elif etype in GATEWAY_TYPES:
                stats["gateways"] += 1
            elif etype in ("startEvent", "endEvent"):
                stats["events"] += 1
        return stats


    @staticmethod
    def _branch_elements(branch: Any) -> List[Any]:

        if isinstance(branch, list):
            return branch
        if isinstance(branch, dict):
            path = branch.get("path", [])
            return path if isinstance(path, list) else []
        return []

    def _iter_elements(self, elements: List[Any]) -> Iterator[Dict[str, Any]]:

        for elem in elements:
            if not isinstance(elem, dict):
                continue
            yield elem
            for branch in elem.get("branches", []) or []:
                yield from self._iter_elements(self._branch_elements(branch))

    def _find_element(self, elements: List[Any], element_id: str) -> Optional[Dict[str, Any]]:
        for elem in self._iter_elements(elements):
            if elem.get("id") == element_id:
                return elem
        return None

    def _normalize_task_types(self, process: List[Any]) -> None:

        for elem in self._iter_elements(process):
            if elem.get("type") == "task":
                elem["type"] = "userTask"

    def _normalize_join_flags(self, process: List[Any]) -> None:

        for elem in self._iter_elements(process):
            if elem.get("type") in GATEWAY_TYPES:
                elem["has_join"] = True


    def extract_tasks(self, bpmn: Dict[str, Any]) -> List[Dict[str, Any]]:

        tasks: List[Dict[str, Any]] = []
        self._traverse(bpmn.get("process", []), 0, None, None, tasks, [])
        return tasks

    def _traverse(
        self,
        elements: List[Any],
        depth: int,
        gateway_id: Optional[str],
        branch_index: Optional[int],
        tasks: List[Dict[str, Any]],
        gateways: List[Dict[str, Any]],
    ) -> None:
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            etype = elem.get("type", "")

            if etype in TASK_TYPES:
                name = elem.get("label", "")
                tasks.append({
                    "id": elem.get("id"),
                    "name": name,
                    "type": etype,
                    "description": elem.get("description") or name,
                    "depth": depth,
                    "gateway_context": {
                        "gateway_id": gateway_id,
                        "branch_index": branch_index,
                        "depth": depth,
                    },
                })

            elif etype in GATEWAY_TYPES:
                branches = elem.get("branches", []) or []
                gateways.append({
                    "id": elem.get("id"),
                    "type": etype.replace("Gateway", ""),
                    "label": elem.get("label", ""),
                    "has_join": elem.get("has_join", True),
                    "branch_count": len(branches),
                    "depth": depth,
                    "conditions": [
                        b["condition"] for b in branches
                        if isinstance(b, dict) and b.get("condition")
                    ],
                })
                for index, branch in enumerate(branches):
                    self._traverse(
                        self._branch_elements(branch), depth + 1,
                        elem.get("id"), index, tasks, gateways,
                    )

    def _extract_enhanced_info(self, bpmn: Dict[str, Any]) -> Dict[str, Any]:
        tasks: List[Dict[str, Any]] = []
        gateways: List[Dict[str, Any]] = []
        self._traverse(bpmn.get("process", []), 0, None, None, tasks, gateways)
        return {
            "task_count": len(tasks),
            "gateway_count": len(gateways),
            "exclusive_count": sum(1 for g in gateways if g["type"] == "exclusive"),
            "parallel_count": sum(1 for g in gateways if g["type"] == "parallel"),
            "inclusive_count": sum(1 for g in gateways if g["type"] == "inclusive"),
            "tasks": tasks,
            "gateways": gateways,
        }

    def _build_condition_set(self, process: List[Any]) -> List[Dict[str, Any]]:

        condition_set = []
        for elem in self._iter_elements(process):
            if elem.get("type") not in CONDITIONAL_GATEWAY_TYPES:
                continue
            for index, branch in enumerate(elem.get("branches", []) or []):
                if isinstance(branch, dict) and branch.get("condition"):
                    condition_set.append({
                        "gateway_id": elem.get("id"),
                        "branch_index": index,
                        "condition": branch["condition"],
                    })
        return condition_set


    def _assemble_output(
        self,
        bpmn: Dict[str, Any],
        validation: BPMNValidationResult,
        is_fallback: bool,
    ) -> Dict[str, Any]:
        bpmn["condition_set"] = self._build_condition_set(bpmn.get("process", []))
        info = self._extract_enhanced_info(bpmn)
        evaluation_format = self._convert_to_evaluation_format(bpmn, info)
        self.stats["total_gateways_detected"] += info["gateway_count"]

        if self.debug:
            print("\n[Statistics]")
            print("  Tasks: %d" % info["task_count"])
            print("  Gateways: %d (exclusive %d, parallel %d, inclusive %d)" % (
                info["gateway_count"], info["exclusive_count"],
                info["parallel_count"], info["inclusive_count"]))

        metadata = {
            "generator": "EnhancedBPMNGenerator",
            "timestamp": datetime.now().isoformat(),
            "is_fallback": is_fallback,
        }
        return {
            "bpmn_workflow": bpmn,
            "evaluation_format": evaluation_format,
            "enhanced_info": info,
            "validation": {
                "is_valid": validation.is_valid,
                "errors": validation.errors,
                "warnings": validation.warnings,
            },
            "metadata": metadata,
        }

    def _convert_to_evaluation_format(
        self, bpmn: Dict[str, Any], info: Dict[str, Any]
    ) -> Dict[str, Any]:

        process = bpmn.get("process", [])
        dependencies = self._build_dependencies(process)

        gateways = []
        for gw in info["gateways"]:
            gw_id = gw["id"]
            element = self._find_element(process, gw_id) or {}
            task_ids = [
                e.get("id") for branch in element.get("branches", []) or []
                for e in self._iter_elements(self._branch_elements(branch))
                if e.get("type") in TASK_TYPES
            ]
            entry_point = next((d["source"] for d in dependencies if d["target"] == gw_id), None)
            join_id = "%s_join" % gw_id
            merge_point = join_id if any(d["target"] == join_id for d in dependencies) else None

            gateway_obj = {
                "id": gw_id,
                "type": gw["type"],
                "tasks": {
                    "tasks": task_ids,
                    "type": gw["type"],
                    "pattern": "fork",
                    "id": gw_id,
                    "entry_point": entry_point,
                },
            }
            if merge_point:
                gateway_obj["tasks"]["merge_point"] = merge_point
            if gw["conditions"]:
                gateway_obj["tasks"]["conditions"] = gw["conditions"]
            gateways.append(gateway_obj)

        return {
            "tasks": [
                {
                    "task_id": t["id"],
                    "name": t["name"],
                    "type": t["type"],
                    "description": t["description"],
                    "gateway_context": t["gateway_context"],
                }
                for t in info["tasks"]
            ],
            "dependencies": dependencies,
            "gateways": gateways,
            "contexts": self._build_gateway_contexts(info["gateways"], dependencies),
            "api_bindings": [],
        }

    def _build_dependencies(self, process: List[Any]) -> List[Dict[str, str]]:
        """Dependency graph; every gateway fork is closed by an explicit join node."""
        dependencies: List[Dict[str, str]] = []
        seen = set()

        def add(source: Optional[str], target: str) -> None:
            if source and target and (source, target) not in seen:
                seen.add((source, target))
                dependencies.append({"source": source, "target": target})

        def build(elements: List[Any], predecessor: Optional[str]) -> Optional[str]:
            prev = predecessor
            for elem in elements:
                if not isinstance(elem, dict) or not elem.get("id"):
                    continue
                elem_id = elem["id"]
                add(prev, elem_id)

                if elem.get("type") in GATEWAY_TYPES:
                    endings = [
                        build(self._branch_elements(branch), elem_id)
                        for branch in elem.get("branches", []) or []
                    ]
                    if endings:
                        join_id = "%s_join" % elem_id
                        for ending in endings:
                            add(ending, join_id)
                        prev = join_id
                    else:
                        prev = elem_id
                else:
                    prev = elem_id
            return prev

        build(process, None)
        return dependencies

    @staticmethod
    def _build_gateway_contexts(
        gateways: List[Dict[str, Any]], dependencies: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:

        in_degree: Dict[str, int] = {}
        out_degree: Dict[str, int] = {}
        predecessors: Dict[str, List[str]] = {}
        successors: Dict[str, List[str]] = {}
        for dep in dependencies:
            source, target = dep["source"], dep["target"]
            out_degree[source] = out_degree.get(source, 0) + 1
            successors.setdefault(source, []).append(target)
            in_degree[target] = in_degree.get(target, 0) + 1
            predecessors.setdefault(target, []).append(source)

        def record(node_id: str, node_type: str) -> Dict[str, Any]:
            return {
                "id": node_id,
                "type": node_type,
                "in_degree": in_degree.get(node_id, 0),
                "out_degree": out_degree.get(node_id, 0),
                "predecessors": predecessors.get(node_id, []),
                "successors": successors.get(node_id, []),
            }

        contexts = []
        for gw in gateways:
            gw_id, gw_type = gw["id"], gw["type"]
            join_id = "%s_join" % gw_id
            contexts.append(record(gw_id, gw_type))
            if join_id in in_degree or join_id in out_degree:
                contexts.append(record(join_id, "%s_join" % gw_type))
        return contexts



    def _generate_fallback(self, reason: str) -> Dict[str, Any]:

        if self.debug:
            print("\n[Fallback] using the sequential template (%s)" % reason)
        bpmn = {
            "process": [
                {"type": "startEvent", "id": "start"},
                {"type": "userTask", "id": "task_1", "label": "Process request",
                 "description": "Process the incoming request."},
                {"type": "serviceTask", "id": "task_2", "label": "Execute action",
                 "description": "Execute the required action."},
                {"type": "userTask", "id": "task_3", "label": "Complete process",
                 "description": "Complete the process."},
                {"type": "endEvent", "id": "end"},
            ]
        }
        validation = BPMNValidationResult(
            True, [], ["Using fallback template: %s" % reason], {}
        )
        return self._assemble_output(bpmn, validation, is_fallback=True)



    def get_statistics(self) -> Dict[str, Any]:
        total = self.stats["total_requests"]
        return {
            **self.stats,
            "success_rate": self.stats["successful_generations"] / total if total else 0,
        }


def create_enhanced_generator(
    api_key: Optional[str] = None, debug: bool = True
) -> EnhancedBPMNGenerator:
    return EnhancedBPMNGenerator(api_key=api_key, debug=debug)
