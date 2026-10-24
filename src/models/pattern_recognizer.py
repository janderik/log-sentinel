
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.features.vectorizer import LogVectorizer


class PatternRecognizer:
    def __init__(self, max_features: int = 1000) -> None:
        self._vectorizer = LogVectorizer(max_features=max_features, ngram_range=(2, 4))
        self._fitted = False
        self._patterns: list[str] = []
        self._embedding_matrix: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, messages: list[str]) -> "PatternRecognizer":
        self._patterns = messages.copy()
        self._embedding_matrix = self._vectorizer.fit_transform(messages)
        self._fitted = True
        return self

    def query(self, message: str, top_k: int = 5) -> list[dict]:
        if not self._fitted:
            raise RuntimeError("PatternRecognizer not fitted. Call fit() first.")
        vec = self._vectorizer.transform([message])
        sims = cosine_similarity(vec, self._embedding_matrix)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results: list[dict] = []
        for idx in top_indices:
            if sims[idx] > 0:
                results.append({
                    "index": int(idx),
                    "message": self._patterns[idx],
                    "similarity": float(round(sims[idx], 4)),
                })
        return results

    def find_clusters(self, threshold: float = 0.7) -> list[list[int]]:
        if not self._fitted:
            raise RuntimeError("PatternRecognizer not fitted. Call fit() first.")
        sim_matrix = cosine_similarity(self._embedding_matrix)
        visited = set()
        clusters: list[list[int]] = []
        for i in range(len(self._patterns)):
            if i in visited:
                continue
            cluster = [j for j in range(len(self._patterns))
                       if j not in visited and sim_matrix[i, j] >= threshold]
            visited.update(cluster)
            if cluster:
                clusters.append(cluster)
        return clusters

    def save(self, path: str) -> None:
        import joblib
        data = {
            "vectorizer": self._vectorizer,
            "patterns": self._patterns,
            "embedding_matrix": self._embedding_matrix,
        }
        joblib.dump(data, path)

    @classmethod
    def load(cls, path: str) -> "PatternRecognizer":
        import joblib
        data = joblib.load(path)
        inst = cls()
        inst._vectorizer = data["vectorizer"]
        inst._patterns = data["patterns"]
        inst._embedding_matrix = data.get("embedding_matrix")
        inst._fitted = True
        return inst
