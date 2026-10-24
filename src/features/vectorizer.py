
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidfVectorizer


class LogVectorizer:
    def __init__(self, max_features: int = 1000, ngram_range: tuple[int, int] = (1, 3)) -> None:
        self._vectorizer = SkTfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            analyzer="char_wb",
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(self, texts: list[str]) -> "LogVectorizer":
        self._vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            self.fit(texts)
        return self._vectorizer.transform(texts).toarray()

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        result = self._vectorizer.fit_transform(texts).toarray()
        self._fitted = True
        return result

    def get_feature_names(self) -> list[str]:
        return self._vectorizer.get_feature_names_out().tolist()

    def save(self, path: str) -> None:
        joblib.dump(self._vectorizer, path)

    @classmethod
    def load(cls, path: str) -> "LogVectorizer":
        inst = cls()
        inst._vectorizer = joblib.load(path)
        inst._fitted = True
        return inst
