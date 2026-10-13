# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import re
import requests
import traceback
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # evaluation_framework_complete/
EVAL_ROOT = SCRIPT_DIR.parent  # evaluation_framework_complete/evaluation/


POSSIBLE_MODEL_PATHS = [

    PROJECT_ROOT / "models" / "Model_simple" / "Model simple",
    PROJECT_ROOT / "models" / "Model_simple" / "Model_simple",

    PROJECT_ROOT / "Model simple2",
    PROJECT_ROOT / "Model_simple2",

    PROJECT_ROOT.parent / "Model simple2",
    PROJECT_ROOT.parent / "Model_simple2",

    SCRIPT_DIR / "Model simple2",
    SCRIPT_DIR / "Model_simple2",
    SCRIPT_DIR,
]

def find_model_root():

    for path in POSSIBLE_MODEL_PATHS:

        if (path / "complete_main.py").exists():
            print(f"  找到模型目录: {path}")
            return path

        if (path / "data" / "api_name.json").exists():
            print(f"  找到数据目录: {path}")
            return path
    return POSSIBLE_MODEL_PATHS[0]

MODEL_ROOT = find_model_root()


sys.path.insert(0, str(MODEL_ROOT))
sys.path.insert(0, str(EVAL_ROOT / "common"))

from scenarios import BPMN_TEST_SCENARIOS, UnifiedOutput, save_results




class LLMClient:

    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = "https://api.deepseek.com/v1"
    
    def chat(self, prompt: str, system: str = None, max_retries: int = 3) -> str:

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(max_retries):
            try:
                print(f"    正在调用DeepSeek API... (尝试 {attempt + 1}/{max_retries})")
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": messages,
                        "max_tokens": 2500,
                        "temperature": 0.3
                    },
                    timeout=180
                )
                if response.status_code == 200:
                    print(f"    ✓ API调用成功")
                    return response.json()["choices"][0]["message"]["content"]
                else:
                    print(f"    API Error: HTTP {response.status_code}")
            except requests.exceptions.Timeout:
                print(f"    ⚠ LLM Timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    print(f"      等待10秒后重试...")
                    time.sleep(10)
            except requests.exceptions.ConnectionError as e:
                print(f"    ⚠ 网络连接错误")
                if attempt < max_retries - 1:
                    print(f"      等待15秒后重试...")
                    time.sleep(15)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"    LLM Error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
        
        print(f"    ✗ API调用失败，已达最大重试次数")
        return "{}"



class ModelSimple2Adapter:

    
    def __init__(self, data_path: str = None, embedding_path: str = None):

        print("\n" + "=" * 60)
        print("初始化 Model_simple2 (Complete Workflow Generation v3.0)")
        print("=" * 60)
        

        self.data_path = Path(data_path) if data_path else MODEL_ROOT / "data"
        self.embedding_path = Path(embedding_path) if embedding_path else MODEL_ROOT / "embeddings"
        

        self.api_name_file = self.data_path / "api_name.json"
        self.api_desc_file = self.data_path / "api_description.json"
        self.api_embed_file = self.embedding_path / "api_features_roberta.npy"
        
        self.has_data = self._check_data_files()
        

        self.system = None
        if self.has_data:
            self._init_system()
        else:
            print("\n⚠ 数据文件不完整，将使用LLM-only模式")
            self._init_llm_only_mode()
    
    def _check_data_files(self) -> bool:

        print("\n[数据检查]")
        files = {
            "api_name.json": self.api_name_file,
            "api_description.json": self.api_desc_file,
            "api_features_roberta.npy": self.api_embed_file
        }
        
        all_exist = True
        for name, path in files.items():
            exists = path.exists()
            icon = "✓" if exists else "✗"
            print(f"  {icon} {name}: {path}")
            if not exists:
                all_exist = False
        
        return all_exist
    
    def _init_system(self):

        try:
            from complete_main import CompleteWorkflowSystem, Config
            

            Config.DATA_PATH = self.data_path
            Config.EMBEDDINGS_PATH = self.embedding_path
            Config.API_NAME_FILE = self.api_name_file
            Config.API_DESCRIPTION_FILE = self.api_desc_file
            Config.API_EMBEDDINGS_FILE = self.api_embed_file

            ids_file = self.embedding_path / "api_name.json"
            if ids_file.exists():
                Config.API_IDS_FILE = ids_file
            

            self.system = CompleteWorkflowSystem(debug=False)
            print("\n✓ Model_simple2 Complete Workflow System 初始化成功")
        except Exception as e:
            print(f"\n✗ 初始化失败: {e}")
            traceback.print_exc()
            self._init_llm_only_mode()
    
    def _init_llm_only_mode(self):

        print("\n[LLM-only 模式]")
        self.llm_client = LLMClient()
    
    def run(self, user_request: str, scenario_name: str) -> UnifiedOutput:

        print(f"\n{'='*60}")
        print(f"处理场景: {scenario_name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        if self.system and self.has_data:
            return self._run_full_system(user_request, scenario_name, start_time)
        else:
            return self._run_llm_only(user_request, scenario_name, start_time)
    
    def _run_full_system(self, user_request: str, scenario_name: str, start_time: float) -> UnifiedOutput:

        try:

            result = self.system.process(user_request, output_file=None)
            

            evaluation_format = result.get('evaluation_format', {})
            api_bindings = result.get('api_bindings', {})
            enhanced_info = result.get('enhanced_info', {})
            

            tasks = []
            for task in evaluation_format.get('tasks', []):
                tasks.append({
                    "id": task.get('task_id', ''),
                    "name": task.get('name', ''),
                    "type": task.get('type', 'userTask')
                })
            

            dependencies = evaluation_format.get('dependencies', [])
            

            gateways = []
            for gateway in evaluation_format.get('gateways', []):
                gateways.append({
                    "id": gateway.get('id', ''),
                    "type": gateway.get('type', ''),
                    "tasks": gateway.get('tasks', [])
                })
            

            formatted_api_bindings = {}
            task_name_map = {t["id"]: t["name"] for t in tasks}
            
            for task_id, binding in api_bindings.items():
                if isinstance(binding, dict):

                    raw_candidates = binding.get('top_candidates', [])
                    

                    candidates = []

                    best_api = {
                        "api_name": binding.get('api_name', 'manual-task'),
                        "api_id": binding.get('api_id', ''),
                        "score": binding.get('confidence', 0.5)
                    }
                    candidates.append(best_api)

                    for cand in raw_candidates:
                        if cand.get('api_name') != best_api['api_name']:
                            candidates.append({
                                "api_name": cand.get('api_name', ''),
                                "api_id": cand.get('api_id', ''),
                                "score": cand.get('score', 0.0)
                            })
                        if len(candidates) >= 5:
                            break
                    
                    formatted_api_bindings[task_id] = {
                        "task_name": task_name_map.get(task_id, ''),
                        "api_name": binding.get('api_name', 'manual-task'),
                        "confidence": binding.get('confidence', 0.5),
                        "recommendation_method": binding.get('recommendation_method', 'unknown'),
                        "top5_candidates": candidates
                    }
            
            execution_time = time.time() - start_time
            
            return UnifiedOutput(
                method="Model_simple2_v3.0",
                scenario_name=scenario_name,
                user_request=user_request,
                tasks=tasks,
                dependencies=dependencies,
                gateways=gateways,
                api_bindings=formatted_api_bindings,
                roles=[],
                execution_time=execution_time,
                extra_info={
                    "mode": "complete_workflow_system",
                    "task_count": enhanced_info.get('task_count', len(tasks)),
                    "gateway_count": enhanced_info.get('gateway_count', len(gateways)),
                    "exclusive_count": enhanced_info.get('exclusive_count', 0),
                    "parallel_count": enhanced_info.get('parallel_count', 0),
                    "inclusive_count": enhanced_info.get('inclusive_count', 0),
                    "api_recommendation_enabled": len(api_bindings) > 0
                }
            )
            
        except Exception as e:
            print(f"  ✗ 完整系统执行失败: {e}")
            traceback.print_exc()
            return self._run_llm_only(user_request, scenario_name, start_time)
    
    def _run_llm_only(self, user_request: str, scenario_name: str, start_time: float) -> UnifiedOutput:

        print("[LLM-only模式] 使用LLM模拟Complete Workflow Generation流程")
        

        print("\n[1/2] 生成完整BPMN工作流...")
        workflow_prompt = f"""Generate a complete BPMN workflow for the following process description.

Process Description:
{user_request}

Provide a structured workflow with:
1. Tasks (task_id, name, type)
2. Dependencies (from, to)
3. Gateways (id, type, tasks involved)

Output as JSON:
{{
    "tasks": [{{"task_id": "t1", "name": "...", "type": "userTask"}}, ...],
    "dependencies": [{{"from": "t1", "to": "t2"}}, ...],
    "gateways": [{{"id": "g1", "type": "exclusive", "tasks": ["t2", "t3"]}}, ...]
}}"""

        workflow_response = self.llm_client.chat(workflow_prompt)
        workflow_data = self._parse_json_object(workflow_response)
        
        tasks = workflow_data.get('tasks', [])
        dependencies = workflow_data.get('dependencies', [])
        gateways = workflow_data.get('gateways', [])
        
        print(f"  ✓ 生成 {len(tasks)} 个任务, {len(dependencies)} 个依赖, {len(gateways)} 个网关")
        

        print("\n[2/2] API推荐（Top-5候选）...")
        api_prompt = f"""You are an API matching expert. For each task, suggest the TOP 5 most relevant APIs ranked by relevance.

Tasks:
{json.dumps(tasks, indent=2)}

For each task_id, provide exactly 5 candidate APIs ranked from most to least relevant.
Use real, widely-known API names (e.g., Stripe, Twilio, SendGrid, Google Maps, AWS S3, etc.).

Output ONLY valid JSON, no explanation:
{{
    "t1": [
        {{"api_name": "BestAPI", "score": 0.92}},
        {{"api_name": "SecondAPI", "score": 0.80}},
        {{"api_name": "ThirdAPI", "score": 0.70}},
        {{"api_name": "FourthAPI", "score": 0.60}},
        {{"api_name": "FifthAPI", "score": 0.50}}
    ],
    ...
}}"""

        api_response = self.llm_client.chat(api_prompt)
        api_bindings_raw = self._parse_json_object(api_response)


        api_bindings = {}
        task_name_map = {t.get('task_id', ''): t.get('name', '') for t in tasks}
        for task_id, candidates_raw in api_bindings_raw.items():

            if isinstance(candidates_raw, dict):
                candidates_raw = [{"api_name": candidates_raw.get('api_name', ''), "score": candidates_raw.get('confidence', 0.5)}]
            if isinstance(candidates_raw, list) and len(candidates_raw) > 0:
                top5 = candidates_raw[:5]
                best = top5[0]
                api_bindings[task_id] = {
                    "task_name": task_name_map.get(task_id, ''),
                    "api_name": best.get('api_name', 'manual-task'),
                    "confidence": best.get('score', 0.5),
                    "recommendation_method": "llm_only",
                    "top5_candidates": [
                        {
                            "api_name": c.get('api_name', ''),
                            "api_id": "",
                            "score": c.get('score', 0.0)
                        }
                        for c in top5
                    ]
                }
        print(f"  ✓ 生成 {len(api_bindings)} 个节点的API绑定（每节点Top-5）")
        
        execution_time = time.time() - start_time
        

        formatted_tasks = []
        for task in tasks:
            formatted_tasks.append({
                "id": task.get('task_id', ''),
                "name": task.get('name', ''),
                "type": task.get('type', 'userTask')
            })
        
        return UnifiedOutput(
            method="Model_simple2_v3.0",
            scenario_name=scenario_name,
            user_request=user_request,
            tasks=formatted_tasks,
            dependencies=dependencies,
            gateways=gateways,
            api_bindings=api_bindings,
            roles=[],
            execution_time=execution_time,
            extra_info={
                "mode": "llm_only",
                "note": "Data files not available, using LLM simulation"
            }
        )
    
    def _parse_json_array(self, text: str) -> list:

        try:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        return []
    
    def _parse_json_object(self, text: str) -> dict:

        try:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        return {}



def main():

    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 3 + "Model_simple2 评估运行脚本 (v3.0 Complete)" + " " * 4 + "║")
    print("║" + " " * 6 + "Complete Workflow Generation System" + " " * 9 + "║")
    print("╚" + "═" * 58 + "╝")
    

    data_path = SCRIPT_DIR / "data"
    embedding_path = SCRIPT_DIR / "embedding"
    

    if not data_path.exists():
        data_path = MODEL_ROOT / "data"
    if not embedding_path.exists():
        embedding_path = MODEL_ROOT / "embeddings"
    

    adapter = ModelSimple2Adapter(
        data_path=str(data_path),
        embedding_path=str(embedding_path)
    )
    

    print("\n" + "=" * 60)
    print("开始评估测试场景")
    print("=" * 60)
    
    results = []
    for i, scenario in enumerate(BPMN_TEST_SCENARIOS):
        print(f"\n>>> 场景 [{i+1}/{len(BPMN_TEST_SCENARIOS)}]: {scenario.name}")
        try:
            result = adapter.run(scenario.requirement, scenario.name)
            results.append(result)
            
            print(f"\n结果摘要:")
            print(f"  - 任务数: {len(result.tasks)}")
            print(f"  - 依赖数: {len(result.dependencies)}")
            print(f"  - 网关数: {len(result.gateways)}")
            print(f"  - API绑定数: {len(result.api_bindings)}")
            print(f"  - 执行时间: {result.execution_time:.2f}s")
        except KeyboardInterrupt:
            print(f"\n  ⚠ 用户中断，跳过场景 {scenario.name}")
            print(f"    (再次按Ctrl+C退出整个程序)")
            try:
                time.sleep(2)
            except KeyboardInterrupt:
                print(f"\n程序被用户终止")
                break
            continue
        except Exception as e:
            print(f"  ✗ 场景 {scenario.name} 处理失败: {e}")
            traceback.print_exc()
            continue
    

    output_dir = EVAL_ROOT / "results"
    save_results(results, str(output_dir), "model_simple2")
    
    print("\n" + "=" * 60)
    print("Model_simple2 评估完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
