import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm


class SimCSEEncoder:


    DEFAULT_MODEL_NAME = "princeton-nlp/unsup-simcse-roberta-base"
    EMBEDDING_DIM = 768
    MAX_LENGTH = 512
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_TOP_K = 30


    TASK_TYPE_PREFIXES = {
        "serviceTask": "Automated API task:",
        "userTask": "Manual user task:",
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: Optional[int] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ):

        self.model_name = model_name or self.DEFAULT_MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length or self.MAX_LENGTH
        self._cache_dir = cache_dir
        self._local_files_only = local_files_only
        self._tokenizer = None
        self._model = None
        self._model_loaded = False



    def _load_model(self) -> None:

        if self._model_loaded:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Please install transformers: pip install transformers torch") from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self._cache_dir,
            local_files_only=self._local_files_only,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self._cache_dir,
            local_files_only=self._local_files_only,
        )
        self._model.to(self.device)
        self._model.eval()
        self._model_loaded = True

    @property
    def tokenizer(self):
        self._load_model()
        return self._tokenizer

    @property
    def model(self):
        self._load_model()
        return self._model



    def encode(self, text: str) -> np.ndarray:

        return self.encode_batch([text])[0]

    def encode_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize: bool = False,
    ) -> np.ndarray:

        if not texts:
            return np.empty((0, self.EMBEDDING_DIM), dtype=np.float32)

        batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        starts = range(0, len(texts), batch_size)
        if show_progress:
            starts = tqdm(starts, desc="Encoding", unit="batch")

        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for start in starts:
                inputs = self.tokenizer(
                    texts[start:start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                embeddings = self.model(**inputs).last_hidden_state[:, 0, :]
                if normalize:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                outputs.append(embeddings.cpu().numpy())

        return np.vstack(outputs)



    def build_task_text(self, task_spec: Dict) -> str:

        description = task_spec.get("description") or task_spec.get(
            "task_name", task_spec.get("name", "")
        )
        prefix = self.TASK_TYPE_PREFIXES.get(task_spec.get("type", ""))
        return "%s %s" % (prefix, description) if prefix else description

    def encode_task_specification(self, task_spec: Dict) -> np.ndarray:

        return self.encode(self.build_task_text(task_spec))



    @staticmethod
    def _build_api_text(api_info: Dict) -> str:

        parts = []
        if api_info.get("name"):
            parts.append(api_info["name"])
        if api_info.get("description"):
            parts.append(api_info["description"])
        if api_info.get("category"):
            parts.append("Category: %s" % api_info["category"])
        tags = api_info.get("tags")
        if tags:
            if isinstance(tags, list):
                parts.append("Tags: %s" % ", ".join(tags[:3]))
            else:
                parts.append("Tags: %s" % tags)
        return " ".join(parts)

    def build_api_matrix(
        self,
        api_database: Dict[str, Dict],
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> Tuple[List[str], np.ndarray]:

        ids = list(api_database.keys())
        texts = [self._build_api_text(api_database[i]) for i in ids]
        matrix = self.encode_batch(texts, batch_size=batch_size, show_progress=show_progress)
        return ids, matrix

    def add_api_embedding(
        self,
        api_id: str,
        api_info: Dict,
        ids: List[str],
        matrix: np.ndarray,
    ) -> Tuple[List[str], np.ndarray]:

        embedding = self.encode(self._build_api_text(api_info)).reshape(1, -1)
        new_matrix = embedding if matrix.size == 0 else np.vstack([matrix, embedding])
        return list(ids) + [api_id], new_matrix

    def precompute_api_embeddings(
        self,
        api_database: Dict[str, Dict],
        save_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:

        ids, matrix = self.build_api_matrix(api_database, batch_size, show_progress)
        if save_path:
            self.save_api_matrix(ids, matrix, save_path)
        return {api_id: matrix[i] for i, api_id in enumerate(ids)}

    @staticmethod
    def save_api_matrix(ids: List[str], matrix: np.ndarray, save_path: str) -> None:

        path = Path(save_path)
        if path.suffix != ".npy":
            raise ValueError("Embeddings are saved as .npy, got: %s" % path.suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
        with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(ids, f)

    @staticmethod
    def load_api_matrix(load_path: str) -> Tuple[List[str], np.ndarray]:

        path = Path(load_path)
        matrix = np.load(path)
        with open(path.with_suffix(".json"), "r", encoding="utf-8") as f:
            ids = json.load(f)
        return ids, matrix

    def load_embeddings(self, load_path: str) -> Dict[str, np.ndarray]:

        ids, matrix = self.load_api_matrix(load_path)
        return {api_id: matrix[i] for i, api_id in enumerate(ids)}



    def compute_similarity(
        self,
        query_embedding: np.ndarray,
        target_embeddings,
        top_k: Optional[int] = None,
        target_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, float]]:

        top_k = top_k or self.DEFAULT_TOP_K

        if isinstance(target_embeddings, dict):
            ids = list(target_embeddings.keys())
            matrix = np.array([target_embeddings[i] for i in ids])
        elif isinstance(target_embeddings, np.ndarray):
            matrix = target_embeddings
            ids = target_ids if target_ids is not None else [str(i) for i in range(len(matrix))]
        else:
            raise ValueError("Unsupported target_embeddings type: %s" % type(target_embeddings))

        query = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        targets = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
        scores = targets @ query

        order = np.argsort(scores)[::-1][:top_k]
        return [(ids[i], float(scores[i])) for i in order]



    def get_embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    def to(self, device: str) -> "SimCSEEncoder":

        self.device = device
        if self._model_loaded:
            self._model.to(device)
        return self


def create_encoder(
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> SimCSEEncoder:

    return SimCSEEncoder(model_name=model_name, device=device, **kwargs)


if __name__ == "__main__":
    encoder = SimCSEEncoder()

    api_database = {
        "api_001": {"name": "OpenWeather API",
                    "description": "Provides weather data and forecasts",
                    "category": "Weather", "tags": ["weather", "forecast", "climate"]},
        "api_002": {"name": "Google Maps API",
                    "description": "Mapping and navigation services",
                    "category": "Mapping", "tags": ["maps", "navigation", "location"]},
        "api_003": {"name": "ShipRate API",
                    "description": "Shipping rate comparison across logistics carriers",
                    "category": "Shipping", "tags": ["shipping", "logistics", "rates"]},
    }
    ids, matrix = encoder.build_api_matrix(api_database, show_progress=False)
    print("U_a shape:", matrix.shape)

    task = {"type": "serviceTask",
            "description": "obtain shipping quote (logistics carrier API)"}
    print("Encoder input:", encoder.build_task_text(task))

    query = encoder.encode_task_specification(task)
    for api_id, score in encoder.compute_similarity(query, matrix, top_k=3, target_ids=ids):
        print("  %s: %.4f" % (api_id, score))
