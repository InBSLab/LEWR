import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

COMMON_DIR = Path(__file__).parent.absolute()
GROUND_TRUTH_DIR = COMMON_DIR.parent.parent / "ground_truth"


@dataclass
class BPMNTestScenario:


    name: str
    requirement: str                 # process description text fed to every method
    process_name: str                # human-readable process title (from ground truth)
    lanes: List[str]                 # participant/lane names in the reference model
    ground_truth_file: str           # relative path under ground_truth/, e.g. "01-dispatch-of-goods/solution.bpmn"
    ground_truth_summary: Dict[str, int]  # task/gateway/event/flow counts, from parsing the file above
    domain: str = ""                 # short domain tag, used for API-retrieval query construction

    def load_ground_truth(self) -> Dict:

        from tools.extract_ground_truth import extract_bpmn_structure
        path = GROUND_TRUTH_DIR / self.ground_truth_file
        return extract_bpmn_structure(str(path))

    def load_gold_api_bindings(self) -> List[Dict]:

        gold_path = GROUND_TRUTH_DIR / "api_gold.json"
        with open(gold_path, "r", encoding="utf-8") as f:
            gold = json.load(f)
        return gold[self.name]["tasks"]




BPMN_TEST_SCENARIOS: List[BPMNTestScenario] = [

    BPMNTestScenario(
        name="dispatch",
        process_name="Dispatch of Goods (Computer Hardware Shop)",
        domain="logistics",
        lanes=["Secretary", "Logistics", "Warehouse"],
        requirement=(
            "If goods shall be shipped, the secretary clarifies who will do the "
            "shipping. If you have large amounts, special shipping will be "
            "necessary. In these cases the secretary invites three logistics "
            "companies to make offers and she selects one of them. In case of "
            "small amounts, normal post shipment is used. Therefore a package "
            "label is written by the secretary and a parcel insurance is taken "
            "by the logistics department head if necessary. In the meantime the "
            "goods can already be packaged by the warehousemen. If everything is "
            "ready, the packaged goods are prepared for being picked up by the "
            "logistics company.\n\n"
            "Background: this process happens at a small hardware company that "
            "ships small amounts of goods to end customers but also big amounts "
            "to other shops."
        ),
        ground_truth_file="01-dispatch-of-goods/solution.bpmn",

        ground_truth_summary={"tasks": 7, "gateways": 6, "events": 2, "flows": 17},
    ),

    BPMNTestScenario(
        name="recourse",
        process_name="Recourse (Insurance Clerk)",
        domain="insurance",
        lanes=["Clerk"],
        requirement=(
            "If an insurant could possibly be subrogated against, I get "
            "information about that. I check that case and if the possibility "
            "is really there, I send a request for payment to the insurant and "
            "make myself a reminder. If recourse is not possible, I close the "
            "case. When we receive the money, I make a booking and close the "
            "case. If the insurant disagrees with the recourse, I have to check "
            "the reasoning of that. If he is right, I simply close the case. If "
            "he is wrong, I forward the case to a collection agency. If the "
            "deadline for disagreement is reached and we haven't received any "
            "money, I forward the case to the collection agency as well.\n\n"
            "Background: insurants can be forced to pay back money they "
            "received from the insurance company for different reasons. This "
            "is called recourse."
        ),
        ground_truth_file="02-recourse/solution.bpmn",

        ground_truth_summary={"tasks": 9, "gateways": 3, "events": 8, "flows": 20},
    ),

    BPMNTestScenario(
        name="credit",
        process_name="Credit Scoring (Bank / Credit Protection Agency)",
        domain="finance",
        lanes=["Banking Frontend", "Scoring (Bank)", "Scoring (Credit Protection Agency)"],
        requirement=(
            "A credit protection agency allows customers to query a credit "
            "rating for persons via a technical interface. The sales clerks in "
            "a bank can use their software frontend to receive the credit "
            "scoring for a certain customer. This starts a process in the "
            "banking system which communicates with the agency in the "
            "background. This process sends a scoring request to the agency "
            "right after the beginning. Then the agency does a first quick "
            "scoring (level 1). This will often lead to an immediate result "
            "which is then returned directly to the banking system within "
            "seconds; the banking process presents the result to the clerk. "
            "Sometimes the scoring cannot be determined immediately and takes "
            "longer. In this case the agency informs the banking process of "
            "the delay and then starts the level 2 scoring (which can take up "
            "to a couple of minutes). After the scoring result is determined, "
            "the information is sent back to the banking process, which "
            "displays a message to the clerk when he receives information "
            "about the delay, and shows the result at the frontend as soon as "
            "it arrives.\n\n"
            "Model with 3 pools: Banking Frontend (collapsed), "
            "Scoring (Bank) (expanded), Scoring (Credit Protection Agency) "
            "(expanded)."
        ),
        ground_truth_file="03-credit-scoring/solution.bpmn",

        ground_truth_summary={"tasks": 7, "gateways": 4, "events": 5, "flows": 16},
    ),

    BPMNTestScenario(
        name="restaurant",
        process_name="Self-Service Restaurant (Optimised Process)",
        domain="hospitality",
        lanes=["Guest", "Employee", "Chef"],
        requirement=(
            "A self-service restaurant is under chaotic conditions. Guests "
            "place their order at the till and receive their meals on call "
            "from the kitchen. As the restaurant is very popular, the process "
            "needs to be adapted to the increasing visitor numbers. In future, "
            "guests should only be in touch with one member of staff for their "
            "order; the chef should purely concentrate on preparing meals. "
            "Buzzers will be introduced to signal to customers when their "
            "order has been completed.\n\n"
            "Optimised process: a guest enters the restaurant when feeling "
            "hungry, chooses a dish from the changing meal range, and waits "
            "until it is his turn. He then places his order with the "
            "employee. The employee enters the order into the POS system and "
            "collects the money from the guest. After payment, the employee "
            "sets up a buzzer and passes it to the guest with the information "
            "\"when the buzzer rings, your dinner is ready\". The employee then "
            "informs the chef of the new order. The chef prepares the meal "
            "and places it in the service hatch, then informs the employee "
            "that the meal is ready. As soon as the employee knows the meal is "
            "ready, he sets off the guest's buzzer. The guest picks up the "
            "meal at the hatch and eats it. Should a guest not react to the "
            "buzzer, the employee calls for him after 5 minutes, if "
            "necessary several times in a row.\n\n"
            "Model with 3 pools: Guest (food consumption), "
            "Employee (order processing), Chef (meal preparation)."
        ),
        ground_truth_file="04-self-service-restaurant/solution.bpmn",

        ground_truth_summary={"tasks": 18, "gateways": 1, "events": 11, "flows": 28},
    ),
]



@dataclass
class UnifiedOutput:


    method: str
    scenario_name: str
    user_request: str
    tasks: List[Dict] = field(default_factory=list)
    dependencies: List[Dict] = field(default_factory=list)
    gateways: List[Dict] = field(default_factory=list)
    api_bindings: Dict[str, Dict] = field(default_factory=dict)
    roles: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    extra_info: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "method": self.method,
            "scenario_name": self.scenario_name,
            "user_request": self.user_request,
            "tasks": self.tasks,
            "dependencies": self.dependencies,
            "gateways": self.gateways,
            "api_bindings": self.api_bindings,
            "roles": self.roles,
            "execution_time": self.execution_time,
            "extra_info": self.extra_info,
        }


def save_results(results: List[UnifiedOutput], output_dir: str, tag: str) -> str:

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}_all_results.json"
    payload = [r.to_dict() for r in results]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} result(s) to {out_path}")
    return str(out_path)


if __name__ == "__main__":

    for s in BPMN_TEST_SCENARIOS:
        print(f"[{s.name}] {s.process_name} — lanes={s.lanes}")
        print(f"    ground truth: {s.ground_truth_summary}")
        print(f"    requirement ({len(s.requirement)} chars): {s.requirement[:80]}...")
