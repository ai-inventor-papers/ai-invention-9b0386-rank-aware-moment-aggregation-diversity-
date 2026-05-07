"""Text embedder using GloVe via sentence-transformers."""

from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor


class GloveTextEmbedding:
    def __init__(self, device: Optional[torch.device] = None):
        self.model = SentenceTransformer(
            "sentence-transformers/average_word_embeddings_glove.6B.300d",
            device=device,
        )

    def __call__(self, sentences: List[str]) -> Tensor:
        cleaned: List[str] = []
        for s in sentences:
            if s is None:
                cleaned.append("")
            elif isinstance(s, str):
                cleaned.append(s)
            else:
                try:
                    if isinstance(s, float) and s != s:  # NaN
                        cleaned.append("")
                    else:
                        cleaned.append(str(s))
                except Exception:
                    cleaned.append("")
        return self.model.encode(cleaned, convert_to_tensor=True)
