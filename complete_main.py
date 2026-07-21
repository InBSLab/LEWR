import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

# 导入完整的工作流生成器
from complete_workflow_generator import CompleteWorkflowGenerator


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
    TOP_K_APIS = 10
    LLM_RERANK_THRESHOLD = 0.4
    
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
        print("Complete Workflow Generation System v3.0")
        print("=" * 70)
        print("  Features:")
        print("    ✓ BPMN Workflow Generation (Enhanced)")
        print("    ✓ API Recommendation (Vector + LLM)")
        print("    ✓ Task Enhancement (Semantic Mapping)")
        print("=" * 70)
        

        api_files = Config.check_api_files()
        print("\n[API Files Status]")
        for name, exists in api_files.items():
            status = "✓" if exists else "✗"
            print(f"  {status} {name}")
        

        enable_api = Config.ENABLE_API_RECOMMENDATION and all(api_files.values())
        
        if not enable_api:
            print("\n⚠️ API recommendation disabled (missing API files)")
            print("   System will only generate BPMN workflow")
        

        self.generator = CompleteWorkflowGenerator(
            api_key=api_key or Config.DEEPSEEK_API_KEY,
            api_database_path=Config.API_NAME_FILE if enable_api else None,
            api_embeddings_path=Config.API_EMBEDDINGS_FILE if enable_api else None,
            api_ids_path=Config.API_IDS_FILE if enable_api else None,
            api_description_path=Config.API_DESCRIPTION_FILE if enable_api else None,
            enable_api_recommendation=enable_api,
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
        result["method"] = "Complete Workflow Generation v3.0"
        

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
        """打印结果摘要"""
        print("\n" + "=" * 70)
        print("Generation Complete")
        print("=" * 70)
        
        enhanced_info = result.get("enhanced_info", {})
        api_bindings = result.get("api_bindings", {})
        validation = result.get("validation", {})
        
        print(f"\n[BPMN Statistics]")
        print(f"  Tasks: {enhanced_info.get('task_count', 0)}")
        print(f"  Gateways: {enhanced_info.get('gateway_count', 0)}")
        print(f"    - Exclusive: {enhanced_info.get('exclusive_count', 0)}")
        print(f"    - Parallel: {enhanced_info.get('parallel_count', 0)}")
        print(f"    - Inclusive: {enhanced_info.get('inclusive_count', 0)}")
        print(f"    - EventBased: {enhanced_info.get('eventbased_count', 0)}")
        
        print(f"\n[API Recommendations]")
        print(f"  Total bindings: {len(api_bindings)}")
        

        methods = {}
        for binding in api_bindings.values():
            method = binding.get('recommendation_method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        
        for method, count in methods.items():
            print(f"    - {method}: {count}")
        
        print(f"\n[Validation]")
        print(f"  Valid: {validation.get('is_valid', False)}")
        if validation.get('errors'):
            print(f"  Errors: {validation['errors']}")
        
        print(f"\n[Output Formats]")
        print(f"  ✓ BPMN Workflow (original)")
        print(f"  ✓ Evaluation Format (for assessment)")
        print(f"  ✓ API Bindings (for each task)")
        print(f"  ✓ Complete Output (merged)")
    
    def batch_process(
        self,
        requests: List[str],
        output_prefix: str = "batch_output"
    ) -> List[Dict[str, Any]]:

        results = []
        
        for i, request in enumerate(requests, 1):
            print(f"\n{'='*70}")
            print(f"Batch Processing: {i}/{len(requests)}")
            print(f"{'='*70}")
            
            output_file = f"{output_prefix}_{i}.json"
            result = self.process(request, output_file)
            results.append(result)
        

        self._print_batch_summary(results)
        
        return results
    
    def _print_batch_summary(self, results: List[Dict[str, Any]]):

        print("\n" + "=" * 70)
        print("Batch Processing Summary")
        print("=" * 70)
        
        total_tasks = sum(r["enhanced_info"]["task_count"] for r in results)
        total_gateways = sum(r["enhanced_info"]["gateway_count"] for r in results)
        total_api_bindings = sum(len(r["api_bindings"]) for r in results)
        
        valid_count = sum(1 for r in results if r["validation"]["is_valid"])
        
        print(f"\nProcessed {len(results)} requests:")
        print(f"  Total tasks: {total_tasks}")
        print(f"  Total gateways: {total_gateways}")
        print(f"  Total API bindings: {total_api_bindings}")
        print(f"  Valid outputs: {valid_count}/{len(results)}")



def main():

    parser = argparse.ArgumentParser(
        description="Complete BPMN Workflow Generator with API Recommendation v3.0"
    )
    parser.add_argument(
        "request",
        nargs="?",
        default=None,
        help="Process description (optional, uses default test case if not provided)"
    )
    parser.add_argument(
        "-o", "--output",
        default="complete_output.json",
        help="Output file name (default: complete_output.json)"
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug mode"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch processing on test cases"
    )
    parser.add_argument(
        "--no-api",
        action="store_true",
        help="Disable API recommendation (only generate BPMN)"
    )
    
    args = parser.parse_args()
    

    if args.no_api:
        Config.ENABLE_API_RECOMMENDATION = False
    
    system = CompleteWorkflowSystem(debug=not args.no_debug)
    
    if args.batch:

        test_cases = [
            """If goods shall be shipped, the secretary clarifies who will do the shipping. 
            If you have large amounts, special shipping will be necessary. In these cases 
            the secretary invites three logistic companies to make offers and she selects 
            one of them. In case of small amounts, normal post shipment is used. 
            Therefore a package label is written by the secretary and a parcel insurance 
            taken by the logistics department head if necessary. In the meantime the goods 
            can be already packaged by the warehousemen. If everything is ready, the 
            packaged goods are prepared for being picked up by the logistic company.""",
            
            """A credit institution receives a credit request. Then a simple credit assessment is performed. 
            The request is then reviewed by a clerk. If the loan amount is above 10000 euros, an intensive 
            assessment needs to be performed as well. Based on the result of these examinations, the credit 
            request is then either rejected or accepted. If the credit request is accepted, the acceptance 
            is sent back to the requester.""",
            
            """In a self-service restaurant, guests can select food items in any order. They can choose a 
            starter, a main dish, and a dessert, but the order doesn't matter. Once they have selected all 
            their items, they proceed to checkout and pay."""
        ]
        
        system.batch_process(test_cases, output_prefix="test_case")
    
    else:

        if args.request:
            request = args.request
        else:

            request = """If goods shall be shipped, the secretary clarifies who will do the shipping. 
            If you have large amounts, special shipping will be necessary. In these cases 
            the secretary invites three logistic companies to make offers and she selects 
            one of them. In case of small amounts, normal post shipment is used. 
            Therefore a package label is written by the secretary and a parcel insurance 
            taken by the logistics department head if necessary. In the meantime the goods 
            can be already packaged by the warehousemen. If everything is ready, the 
            packaged goods are prepared for being picked up by the logistic company."""
        

        result = system.process(request, args.output)
        

        print("\n" + "=" * 70)
        print("Sample Output")
        print("=" * 70)
        
        complete_output = result.get("complete_output", {})
        
        print(f"\nTasks with API Recommendations:")
        for task in complete_output.get("tasks", [])[:5]:  # 显示前5个
            task_id = task['task_id']
            task_name = task['name']
            api_rec = task.get('api_recommendation', {})
            api_name = api_rec.get('api_name', 'N/A')
            confidence = api_rec.get('confidence', 0)
            
            print(f"  [{task_id}] {task_name}")
            print(f"      → API: {api_name} (confidence: {confidence:.3f})")
        
        print(f"\nGateways ({len(complete_output.get('gateways', []))}):")
        for gateway in complete_output.get("gateways", []):
            print(f"  [{gateway['id']}] {gateway['type']}")


if __name__ == "__main__":
    main()
