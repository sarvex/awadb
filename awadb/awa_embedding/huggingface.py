from awadb import AwaEmbedding
from typing import Iterable, Any, List

# Use all-mpnet-base-v2 as the default model
DEFAULT_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

class HuggingFaceEmbeddings(AwaEmbedding):
    def __init__(self):
        self.tokenizer = None
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence_transformers`."
            ) from exc
        self.model = SentenceTransformer(DEFAULT_MODEL_NAME)

    def Embedding(self, sentence):
        tokens = []
        if self.tokenizer is None:
            tokens.append(sentence)
        else:
            tokens = self.tokenizer.tokenize(sentence)
        return self.model.encode(tokens[0])

    def EmbeddingBatch(
        self,
        texts: Iterable[str],
        **kwargs: Any,
    ) -> List[List[float]]:
        results: List[List[float]] = [self.model.encode(text) for text in texts]
        return results
