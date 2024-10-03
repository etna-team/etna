from etna import SETTINGS
from etna.transforms.embeddings.models.base import BaseEmbeddingModel

if SETTINGS.torch_required:
    from etna.transforms.embeddings.models.ts2vec import TS2VecEmbeddingModel
    from etna.transforms.embeddings.models.tstcc import TSTCCEmbeddingModel
