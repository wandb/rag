import os
from typing import Dict, List, Optional

import numpy as np
import safetensors
import safetensors.numpy
import weave
from sentence_transformers import SentenceTransformer

import wandb


class BGERetriever(weave.Model):
    model_name: str
    weave_chunked_dataset_address: str
    _corpus: List[Dict[str, str]] = []
    _index: np.ndarray = None
    _model: SentenceTransformer = None

    def __init__(
        self,
        weave_chunked_dataset_address: str,
        model_name: str,
        index: Optional[np.ndarray] = None,
    ):
        super().__init__(
            weave_chunked_dataset_address=weave_chunked_dataset_address,
            model_name=model_name,
        )
        self._index = index
        self._model = SentenceTransformer(self.model_name)
        self._corpus = [
            dict(row)
            for row in weave.ref(self.weave_chunked_dataset_address).get().rows
        ]

    @classmethod
    def from_wandb_artifact(
        cls, artifact_address: str, weave_chunked_dataset_address: str, model_name: str
    ):
        api = wandb.Api()
        artifact = api.artifact(artifact_address)
        artifact_dir = artifact.download()
        with open(os.path.join(artifact_dir, "index.safetensors"), "rb") as f:
            index = f.read()
        index = safetensors.numpy.load(index)["index"]
        return cls(
            weave_chunked_dataset_address=weave_chunked_dataset_address,
            model_name=model_name,
            index=index,
        )

    def create_index(
        self,
        index_persist_dir: Optional[str] = None,
        artifact_name: Optional[str] = None,
    ):
        self._index = self._model.encode(
            sentences=[row["cleaned_content"] for row in self._corpus],
            normalize_embeddings=True,
        )
        if index_persist_dir:
            os.makedirs(index_persist_dir, exist_ok=True)
            safetensors.numpy.save_file(
                tensor_dict={"index": self._index},
                filename=os.path.join(index_persist_dir, "index.safetensors"),
            )
            if wandb.run and artifact_name:
                artifact_metadata = {
                    "weave_chunked_dataset_address": self.weave_chunked_dataset_address,
                    "model_name": self.model_name,
                }
                artifact = wandb.Artifact(
                    name=artifact_name,
                    type="vector_index",
                    metadata=artifact_metadata,
                )
                artifact.add_dir(local_path=index_persist_dir)
                artifact.save()

    @weave.op()
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_embeddings = self._model.encode([query], normalize_embeddings=True)
        scores = query_embeddings @ self._index.T
        sorted_indices = np.argsort(scores, axis=None)[::-1]
        top_k_indices = sorted_indices[:top_k].tolist()
        retrieved_pages = []
        for idx in top_k_indices:
            retrieved_pages.append(
                {
                    "retrieved_content": self._corpus[idx]["cleaned_content"],
                    "metadata": self._corpus[idx]["metadata"],
                }
            )
        return retrieved_pages

    @weave.op()
    def predict(self, query: str, top_k: int = 5) -> list[dict]:
        return self.search(
            "Generate a representation for this sentence that can be used to retrieve related articles:\n"
            + query,
            top_k,
        )


class BGEImageRetriever(weave.Model):
    model_name: str
    _model: SentenceTransformer = None

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self._model = SentenceTransformer(self.model_name)

    @weave.op()
    def search(
        self, query: str, image_descriptions: List[str], top_k: int = 5
    ) -> list[dict]:
        index = self._model.encode(
            sentences=image_descriptions, normalize_embeddings=True
        )
        query_embeddings = self._model.encode([query], normalize_embeddings=True)
        scores = query_embeddings @ index.T
        sorted_indices = np.argsort(scores, axis=None)[::-1]
        top_k_indices = sorted_indices[:top_k].tolist()
        retrieved_pages = []
        for idx in top_k_indices:
            retrieved_pages.append(
                {
                    "retrieved_image_description": image_descriptions[idx],
                    "image_idx": idx,
                }
            )
        return retrieved_pages

    @weave.op()
    def predict(
        self, query: str, image_descriptions: List[str], top_k: int = 1
    ) -> list[dict]:
        return self.search(
            "Generate a representation for this sentence that can be used to retrieve related articles:\n"
            + query,
            image_descriptions,
            top_k,
        )
