# -*- coding:utf-8 -*-
#!/usr/bin/python3

from sentence_transformers import SentenceTransformer
from transformers import AutoModel
from transformers import AutoTokenizer
from typing import Iterable, Any, List

class LLMEmbedding:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = None 


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

    #set your own llm
    def SetModel(self, model_name):
        self.model = AutoModel.from_pretrained(model_name)

    #set your own tokenizer
    def SetTokenizer(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)



