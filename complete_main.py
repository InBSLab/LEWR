# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# 导入完整的工作流生成器
from complete_workflow_generator import (
    CompleteWorkflowGenerator,
    DEFAULT_TOP_K,
    HIGH_SCORE_THRESHOLD,
    LOW_SCORE_THRESHOLD,
)


# ============================================================================
# 配置
# ============================================================================

class Config:


    PROJECT_ROOT = Path(__file__).parent.absolute()
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    DATA_PATH = PROJECT_ROOT / "data"
    EMBEDDINGS_PATH = PROJECT_ROOT / "embeddings"


    API_NAME_FILE = DATA_PATH / "api_name.json"
    API_DESCRIPTION_FILE = DATA_PATH / "api_description.json"
    API_EMBEDDINGS_FILE = EMBEDDINGS_PATH / "api_features_roberta.npy"
    API_IDS_FILE = EMBEDDINGS_PATH / "api_name.json"


    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


    MAX_RETRIES = 3
    VALIDATE_OUTPUT = True
    DEBUG_MODE = True


    ENABLE_API_RECOMMENDATION = True
    TOP_K_APIS = DEFAULT_TOP_K
    HIGH_SCORE_THRESHOLD = HIGH_SCORE_THRESHOLD
    LOW_SCORE_THRESHOLD = LOW_SCORE_THRESHOLD

    @classmethod
    def ensure_output_dir(cls):

        cls.OUTPUT_DIR.mkdir(exist_ok=True)

    @classmethod
    def check_api_files(cls) -> Dict[str, bool]:

        files = {
            "api_name.json": cls.API_NAME_FILE,
            "api_description.json": cls.API_DESCRIPTION_FILE,
            "api_features_roberta.npy": cls.API_EMBEDDINGS_FILE,
            "api_ids.json": cls.API_IDS_FILE,
        }
        return {name: path.exists() for name, path in files.items()}



class CompleteWorkflowSystem:


    def __init__(self, api_key: str = None, debug: bool = True):

        print("=" * 70)
        print("LEWR: BPMN Workflow Generation + API Recommendation")
        print("=" * 70)
        print("  Pipeline:")
        print("    ✓ Stage 1  LLM-based BPMN generation (single LLM call)")
        print("    ✓ Stage 2  Depth-first workflow task extraction")
        print("    ✓ Stage 3  Task semantic enhancement (cross-space mapping)")
        print("    ✓ Stage 4  SimCSE encoding + top-K retrieval + rule re-ranking")
        print("    ✓ Stage 5  Adaptive LLM confirmation + compatibility-score adjustment")
        print("=" * 70)

        api_key = api_key or Config.DEEPSEEK_API_KEY
        if not api_key:
            raise RuntimeError(
                "未找到 DeepSeek API Key。请先设置环境变量 DEEPSEEK_API_KEY "
                "（PowerShell: $env:DEEPSEEK_API_KEY='你的key'；永久: setx DEEPSEEK_API_KEY 你的key）。"
            )


        api_files = Config.check_api_files()
        print("\n[API Files Status]")
        for name, exists in api_files.items():
            print(f"  {'✓' if exists else '✗'} {name}")


        enable_api = Config.ENABLE_API_RECOMMENDATION and all(api_files.values())

        if not enable_api:
            print("\n⚠️ API recommendation disabled (missing API files)")
            print("   System will only generate BPMN workflow")


        self.generator = CompleteWorkflowGenerator(
            api_key=api_key,
            api_database_path=Config.API_NAME_FILE if enable_api else None,
            api_embeddings_path=Config.API_EMBEDDINGS_FILE if enable_api else None,
            api_ids_path=Config.API_IDS_FILE if enable_api else None,
            api_description_path=Config.API_DESCRIPTION_FILE if enable_api else None,
            enable_api_recommendation=enable_api,
            top_k=Config.TOP_K_APIS,
            high_threshold=Config.HIGH_SCORE_THRESHOLD,
            low_threshold=Config.LOW_SCORE_THRESHOLD,
            debug=debug
        )

        print("\n✓ System initialized successfully")

    def process(
        self,
        user_request: str,
        output_file: Optional[str] = None
    ) -> Dict[str, Any]:

        print("\n" + "=" * 70)
        print("Processing User Request")
        print("=" * 70)
        print(f"Input: {user_request[:200]}...")


        result = self.generator.generate(
            user_request,
            max_retries=Config.MAX_RETRIES,
            validate=Config.VALIDATE_OUTPUT
        )


        result["user_request"] = user_request
        result["method"] = "LEWR (paper-aligned v5.0)"


        if output_file:
            self.save_result(result, output_file)


        self._print_summary(result)

        return result

    def save_result(self, result: Dict[str, Any], output_file: str):

        Config.ensure_output_dir()

        output_path = Path(output_file)
        if not output_path.is_absolute():
            output_path = Config.OUTPUT_DIR / output_file

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_path}")

    def _print_summary(self, result: Dict[str, Any]):

        print("\n" + "=" * 70)
        print("Generation Complete")
        print("=" * 70)

        enhanced_info = result.get("enhanced_info", {})
        api_bindings = result.get("api_bindings", {})
        validation = result.get("validation", {})


        print("\n[BPMN Statistics]")
        print(f"  Tasks: {enhanced_info.get('task_count', 0)}")
        print(f"  Gateways: {enhanced_info.get('gateway_count', 0)}")
        print(f"    - Exclusive (XOR): {enhanced_info.get('exclusive_count', 0)}")
        print(f"    - Parallel  (AND): {enhanced_info.get('parallel_count', 0)}")
        print(f"    - Inclusive (OR) : {enhanced_info.get('inclusive_count', 0)}")

        print("\n[API Recommendations]")
        print(f"  Total bindings: {len(api_bindings)}")


        methods: Dict[str, int] = {}
        for binding in api_bindings.values():
            method = binding.get('recommendation_method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        for method, count in methods.items():
            print(f"    - {method}: {count}")


        conf_stats = result.get("api_recommendation_stats", {}).get("confirmer")
        if conf_stats:
            print(f"  LLM confirmation calls: {conf_stats.get('llm_calls', 0)} "
                  f"(cache hits: {conf_stats.get('cache_hits', 0)})")

        print("\n[Validation]")
        print(f"  Valid: {validation.get('is_valid', False)}")
        if validation.get('errors'):
            print(f"  Errors: {validation['errors']}")

        print("\n[Output Formats]")
        print("  ✓ BPMN Workflow (hierarchical JSON)")
        print("  ✓ Evaluation Format (flattened)")
        print("  ✓ API Bindings (per task)")
        print("  ✓ Complete Output (merged)")

    def batch_process(
        self,
        requests: List[str],
        output_prefix: str = "batch_output"
    ) -> List[Dict[str, Any]]:

        results = []

        for i, request in enumerate(requests, 1):
            print(f"\n{'=' * 70}")
            print(f"Batch Processing: {i}/{len(requests)}")
            print(f"{'=' * 70}")

            output_file = f"{output_prefix}_{i}.json"
            results.append(self.process(request, output_file))


        self._print_batch_summary(results)

        return results

    def _print_batch_summary(self, results: List[Dict[str, Any]]):

        print("\n" + "=" * 70)
        print("Batch Processing Summary")
        print("=" * 70)

        total_tasks = sum(r.get("enhanced_info", {}).get("task_count", 0) for r in results)
        total_gateways = sum(r.get("enhanced_info", {}).get("gateway_count", 0) for r in results)
        total_api_bindings = sum(len(r.get("api_bindings", {})) for r in results)
        valid_count = sum(1 for r in results if r.get("validation", {}).get("is_valid", False))

        print(f"\nProcessed {len(results)} requests:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Total gateways: {total_gateways}")
        print(f"  Total API bindings: {total_api_bindings}")
        print(f"  Valid outputs: {valid_count}/{len(results)}")


# ============================================================================
# 命令行接口
# ============================================================================


DEFAULT_TEST_CASE = """If goods shall be shipped, the secretary clarifies who will do the shipping.
If you have large amounts, special shipping will be necessary. In these cases
the secretary invites three logistic companies to make offers and she selects
one of them. In case of small amounts, normal post shipment is used.
Therefore a package label is written by the secretary and a parcel insurance
taken by the logistics department head if necessary. In the meantime the goods
can be already packaged by the warehousemen. If everything is ready, the
packaged goods are prepared for being picked up by the logistic company."""

BATCH_TEST_CASES = [
    DEFAULT_TEST_CASE,

    """A credit institution receives a credit request. Then a simple credit assessment is performed.
The request is then reviewed by a clerk. If the loan amount is above 10000 euros, an intensive
assessment needs to be performed as well. Based on the result of these examinations, the credit
request is then either rejected or accepted. If the credit request is accepted, the acceptance
is sent back to the requester.""",

    """In a self-service restaurant, guests can select food items in any order. They can choose a
starter, a main dish, and a dessert, but the order doesn't matter. Once they have selected all
their items, they proceed to checkout and pay.""",
]


def main():

    parser = argparse.ArgumentParser(
        description="LEWR: BPMN Workflow Generation with API Recommendation"
    )
    parser.add_argument(
        "request",
        nargs="?",
        default=None,
        help="流程描述（可选，不提供时使用默认示例）"
    )
    parser.add_argument(
        "-o", "--output",
        default="complete_output.json",
        help="输出文件名（默认: complete_output.json）"
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="关闭详细日志"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="对内置示例批量运行"
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="关闭API推荐（只生成BPMN）"
    )

    args = parser.parse_args()


    if args.no_api:
        Config.ENABLE_API_RECOMMENDATION = False

    system = CompleteWorkflowSystem(debug=not args.no_debug)

    if args.batch:
        system.batch_process(BATCH_TEST_CASES, output_prefix="test_case")
        return


    request = args.request if args.request else DEFAULT_TEST_CASE
    result = system.process(request, args.output)


    print("\n" + "=" * 70)
    print("Sample Output")
    print("=" * 70)

    complete_output = result.get("complete_output", {})

    print("\nTasks with API Recommendations:")
    for task in complete_output.get("tasks", [])[:5]:
        api_rec = task.get('api_recommendation', {})
        print(f"  [{task.get('task_id', '?')}] {task.get('name', '')}")
        print(f"      → API: {api_rec.get('api_name', 'N/A')} "
              f"(comp*: {api_rec.get('confidence', 0):.3f}, "
              f"method: {api_rec.get('recommendation_method', 'N/A')})")

    print(f"\nGateways ({len(complete_output.get('gateways', []))}):")
    for gateway in complete_output.get("gateways", []):
        print(f"  [{gateway.get('id', '?')}] {gateway.get('type', '?')}")


if __name__ == "__main__":
    main()
