import os


os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import torch
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import json
import pickle
from tqdm import tqdm


class SimCSEEncoder:

    DEFAULT_MODEL_NAME = "princeton-nlp/unsup-simcse-roberta-base"
    

    EMBEDDING_DIM = 768
    

    MAX_LENGTH = 512
    

    DEFAULT_BATCH_SIZE = 32
    

    USE_GPU = True
    

    
    def __init__(
        self,
        model_name: str = None,
        device: str = None,
        max_length: int = None,
        cache_dir: str = None,
        use_fast_tokenizer: bool = True
    ):

        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        

        if device is None:
            self.device = "cuda" if (self.USE_GPU and torch.cuda.is_available()) else "cpu"
        else:
            self.device = device
        

        self.max_length = max_length or self.MAX_LENGTH
        

        self._model_loaded = False
        self._tokenizer = None
        self._model = None
        self._cache_dir = cache_dir
        self._use_fast_tokenizer = use_fast_tokenizer
        
        print(f"[SimCSEEncoder] 初始化完成")
        print(f"  - 模型: {self.model_name}")
        print(f"  - 设备: {self.device}")
        print(f"  - 最大长度: {self.max_length}")
        print(f"  - 模型将在首次使用时加载")
    
    def _load_model(self):

        if self._model_loaded:
            return
        
        print(f"[SimCSEEncoder] 正在加载模型 {self.model_name}...")
        

        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "请安装transformers库: pip install transformers torch\n"
                "如果使用GPU，还需要安装对应版本的CUDA"
            )
        

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self._cache_dir,
            use_fast=self._use_fast_tokenizer,
            local_files_only=True
        )
        

        self._model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self._cache_dir,
            local_files_only=True
        )
        

        self._model.to(self.device)
        

        self._model.eval()
        
        self._model_loaded = True
        print(f"[SimCSEEncoder] 模型加载完成，使用设备: {self.device}")
    
    @property
    def tokenizer(self):

        if not self._model_loaded:
            self._load_model()
        return self._tokenizer
    
    @property
    def model(self):

        if not self._model_loaded:
            self._load_model()
        return self._model
    
    def encode(self, text: str) -> np.ndarray:

        return self.encode_batch([text])[0]
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = False,
        normalize: bool = False
    ) -> np.ndarray:

        if not texts:
            return np.array([])
        
        batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        all_embeddings = []
        

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding", unit="batch")
        
        with torch.no_grad():
            for start_idx in iterator:
                batch_texts = texts[start_idx:start_idx + batch_size]
                

                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
                

                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                

                outputs = self.model(**inputs)
                

                embeddings = outputs.last_hidden_state[:, 0, :]
                

                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                

                embeddings = embeddings.cpu().numpy()
                all_embeddings.append(embeddings)
        

        all_embeddings = np.vstack(all_embeddings)
        
        return all_embeddings
    
    def precompute_api_embeddings(
        self,
        api_database: Dict[str, Dict],
        save_path: str = None,
        batch_size: int = None,
        show_progress: bool = True
    ) -> Dict[str, np.ndarray]:

        print(f"[SimCSEEncoder] 预计算 {len(api_database)} 个API的embeddings...")
        

        api_texts = []
        api_ids = []
        
        for api_id, api_info in api_database.items():
            api_text = self._build_api_text(api_info)
            api_texts.append(api_text)
            api_ids.append(api_id)
        

        embeddings_matrix = self.encode_batch(
            api_texts,
            batch_size=batch_size,
            show_progress=show_progress
        )
        

        api_embeddings = {
            api_id: embeddings_matrix[i]
            for i, api_id in enumerate(api_ids)
        }
        

        if save_path:
            self._save_embeddings(
                api_embeddings,
                save_path,
                include_metadata=True,
                api_database=api_database
            )
        
        print(f"[SimCSEEncoder] 预计算完成！")
        return api_embeddings
    
    def _build_api_text(self, api_info: Dict) -> str:

        components = []
        
        # 1. API名称 (必需)
        if api_info.get('name'):
            components.append(api_info['name'])
        
        # 2. 描述 (核心信息)
        if api_info.get('description'):
            components.append(api_info['description'])
        
        # 3. 类别 (可选)
        if api_info.get('category'):
            components.append(f"Category: {api_info['category']}")
        
        # 4. 标签 (可选，但不要过多)
        if api_info.get('tags'):
            tags = api_info['tags']
            if isinstance(tags, list):
                # 只取前3个标签，避免噪音
                top_tags = tags[:3]
                components.append(f"Tags: {', '.join(top_tags)}")
            else:
                components.append(f"Tags: {tags}")
        
        # 拼接所有组件
        api_text = " ".join(components)
        
        return api_text
    
    def encode_task_specification(self, task_spec: Dict) -> np.ndarray:

        if task_spec.get('semantic_query'):
            return self.encode(task_spec['semantic_query'])
        

        task_name = task_spec.get('task_name', task_spec.get('name', ''))
        

        task_type = task_spec.get('type', 'serviceTask')
        
        if task_type == 'serviceTask':

            task_text = f"Automated API task: {task_name}"
        elif task_type == 'userTask':

            task_text = f"Manual user task: {task_name}"
        elif task_type == 'scriptTask':

            task_text = f"Script execution task: {task_name}"
        else:

            task_text = f"Automated API task: {task_name}"
        

        if task_spec.get('description'):
            task_text += f". {task_spec['description']}"
        

        if task_spec.get('required_capabilities'):
            caps = task_spec['required_capabilities']
            if isinstance(caps, list):
                task_text += f". Requires: {', '.join(caps)}"
            else:
                task_text += f". Requires: {caps}"
        

        if task_spec.get('category'):
            task_text += f". Category: {task_spec['category']}"
        
        return self.encode(task_text)
    
    def _save_embeddings(
        self,
        embeddings: Dict[str, np.ndarray],
        save_path: str,
        include_metadata: bool = False,
        api_database: Dict = None
    ):

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix == '.pkl':

            save_data = {
                'embeddings': embeddings,
                'model_name': self.model_name,
                'embedding_dim': self.EMBEDDING_DIM
            }
            if include_metadata and api_database:
                save_data['metadata'] = api_database
            
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)
                
        elif save_path.suffix == '.npy':

            ids = list(embeddings.keys())
            matrix = np.array([embeddings[i] for i in ids])
            np.save(save_path, matrix)
            

            id_map_path = save_path.with_suffix('.json')
            with open(id_map_path, 'w', encoding='utf-8') as f:
                json.dump(ids, f)
        else:
            raise ValueError(f"不支持的文件格式: {save_path.suffix}")
        
        print(f"[SimCSEEncoder] Embeddings已保存到: {save_path}")
    
    def load_embeddings(self, load_path: str) -> Dict[str, np.ndarray]:

        load_path = Path(load_path)
        
        if load_path.suffix == '.pkl':
            with open(load_path, 'rb') as f:
                data = pickle.load(f)
            return data['embeddings']
            
        elif load_path.suffix == '.npy':
            matrix = np.load(load_path)
            id_map_path = load_path.with_suffix('.json')
            with open(id_map_path, 'r', encoding='utf-8') as f:
                ids = json.load(f)
            return {id_: matrix[i] for i, id_ in enumerate(ids)}
        else:
            raise ValueError(f"不支持的文件格式: {load_path.suffix}")
    
    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        target_embeddings,
        top_k: int = 10,
        metric: str = "cosine",
        target_ids: List[str] = None
    ) -> List[Tuple[str, float]]:

        if isinstance(target_embeddings, dict):
            ids = list(target_embeddings.keys())
            target_matrix = np.array([target_embeddings[i] for i in ids])
        elif isinstance(target_embeddings, np.ndarray):
            target_matrix = target_embeddings
            if target_ids is not None:
                ids = target_ids
            else:

                ids = [str(i) for i in range(len(target_matrix))]
        else:
            raise ValueError(f"不支持的 target_embeddings 类型: {type(target_embeddings)}")
        
        if metric == "cosine":

            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            target_norms = target_matrix / (np.linalg.norm(target_matrix, axis=1, keepdims=True) + 1e-8)
            scores = np.dot(target_norms, query_norm)
            
        elif metric == "dot":
            scores = np.dot(target_matrix, query_embedding)
            
        elif metric == "euclidean":
            distances = np.linalg.norm(target_matrix - query_embedding, axis=1)
            scores = -distances
        else:
            raise ValueError(f"不支持的度量方式: {metric}")
        

        sorted_indices = np.argsort(scores)[::-1][:top_k]
        results = [(ids[i], float(scores[i])) for i in sorted_indices]
        
        return results
    
    def get_embedding_dim(self) -> int:

        return self.EMBEDDING_DIM
    
    def to(self, device: str) -> "SimCSEEncoder":

        self.device = device
        if self._model_loaded:
            self._model.to(device)
        return self



def create_encoder(
    model_name: str = None,
    device: str = None,
    **kwargs
) -> SimCSEEncoder:

    return SimCSEEncoder(model_name=model_name, device=device, **kwargs)



def _test_encoder():

    print("=" * 60)
    print("SimCSE Encoder 功能测试（优化版）")
    print("=" * 60)
    

    encoder = SimCSEEncoder()
    

    print("\n[测试1] 单文本编码")
    text = "Find weather forecast service for outdoor activities"
    embedding = encoder.encode(text)
    print(f"  输入: {text}")
    print(f"  输出维度: {embedding.shape}")
    print(f"  向量示例: {embedding[:5]}...")
    

    print("\n[测试2] 批量编码")
    texts = [
        "Weather forecast API for hiking",
        "GPS navigation service for forest",
        "Hotel booking system"
    ]
    embeddings = encoder.encode_batch(texts)
    print(f"  输入数量: {len(texts)}")
    print(f"  输出形状: {embeddings.shape}")
    

    print("\n[测试3] 增强的任务编码（优化版）")
    

    task_service = {
        "task_id": "t1",
        "task_name": "Schedule shipment",
        "type": "serviceTask",
        "description": "Arrange logistics for package delivery"
    }
    emb_service = encoder.encode_task_specification(task_service)
    print(f"  ServiceTask编码完成: {emb_service.shape}")
    print(f"    → 会添加前缀: 'Automated API task: Schedule shipment'")
    

    task_user = {
        "task_id": "t2",
        "task_name": "Approve request",
        "type": "userTask",
        "description": "Manager reviews and approves the request"
    }
    emb_user = encoder.encode_task_specification(task_user)
    print(f"  UserTask编码完成: {emb_user.shape}")
    print(f"    → 会添加前缀: 'Manual user task: Approve request'")
    

    similarity = np.dot(emb_service, emb_user) / (
        np.linalg.norm(emb_service) * np.linalg.norm(emb_user)
    )
    print(f"  ServiceTask vs UserTask 相似度: {similarity:.4f}")
    print(f"    → 添加类型前缀后，两种任务更容易区分")
    

    print("\n[测试4] 相似度计算")
    target_embeddings = {
        f"text_{i}": embeddings[i] for i in range(len(texts))
    }
    query_embedding = encoder.encode("Weather service for outdoor")
    results = encoder.compute_similarity(query_embedding, target_embeddings, top_k=3)
    print("  查询: Weather service for outdoor")
    print("  相似度排序:")
    for id_, score in results:
        print(f"    - {texts[int(id_.split('_')[1])]}: {score:.4f}")
    

    print("\n[测试5] API embedding预计算")
    api_database = {
        "api_001": {
            "name": "OpenWeather API",
            "description": "Provides weather data and forecasts",
            "category": "Weather",
            "tags": ["weather", "forecast", "climate"]
        },
        "api_002": {
            "name": "Google Maps API",
            "description": "Mapping and navigation services",
            "category": "Mapping",
            "tags": ["maps", "navigation", "location"]
        }
    }
    api_embeddings = encoder.precompute_api_embeddings(api_database, show_progress=False)
    print(f"  预计算完成: {len(api_embeddings)} 个API")
    
    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
    print("\n优化说明:")
    print("1. 任务类型前缀: 区分serviceTask(API)和userTask(人工)")
    print("2. 提高API匹配准确度: 避免将人工任务错误匹配API")
    print("3. 更丰富的语义信息: 融合类型、描述、能力需求")


if __name__ == "__main__":
    _test_encoder()
