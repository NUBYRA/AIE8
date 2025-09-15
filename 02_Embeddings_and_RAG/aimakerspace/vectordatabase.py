import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable, Dict, Any, Optional
from aimakerspace.openai_utils.embedding import EmbeddingModel
from dotenv import load_dotenv


def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Euclidean distance between two vectors."""
    return np.linalg.norm(vector_a - vector_b)

def manhattan_distance(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the Manhattan distance between two vectors."""
    return np.sum(np.abs(vector_a - vector_b))

def dot_product_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the dot product similarity between two vectors."""
    return np.dot(vector_a, vector_b)

class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None, distance_measure: Callable = cosine_similarity):
        self.vectors = defaultdict(np.array)
        self.metadata = defaultdict(dict)  # Store metadata for each vector
        self.embedding_model = embedding_model or EmbeddingModel()
        self.distance_measure = distance_measure

    def insert(self, key: str, vector: np.array, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Insert a vector with optional metadata."""
        self.vectors[key] = vector
        if metadata:
            self.metadata[key] = metadata

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors with optional metadata filtering."""
        scores = []
        for key, vector in self.vectors.items():
            # Apply metadata filter if provided
            if metadata_filter:
                if not self._matches_filter(key, metadata_filter):
                    continue
            
            score = distance_measure(query_vector, vector)
            metadata = self.metadata.get(key, {})
            scores.append((key, score, metadata))
        
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors by text with optional metadata filtering."""
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure, metadata_filter)
        return [(result[0], result[1], result[2]) for result in results] if not return_as_text else [result[0] for result in results]

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)
    
    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Get metadata for a specific key."""
        return self.metadata.get(key, {})
    
    def _matches_filter(self, key: str, metadata_filter: Dict[str, Any]) -> bool:
        """Check if a key's metadata matches the filter criteria."""
        key_metadata = self.metadata.get(key, {})
        for filter_key, filter_value in metadata_filter.items():
            if filter_key not in key_metadata:
                return False
            if key_metadata[filter_key] != filter_value:
                return False
        return True

    async def abuild_from_list(self, list_of_text: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None) -> "VectorDatabase":
        """Build vector database from list of texts with optional metadata."""
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for i, (text, embedding) in enumerate(zip(list_of_text, embeddings)):
            metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
            self.insert(text, np.array(embedding), metadata)
        return self


