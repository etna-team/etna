import torch

from etna.models.nn.utils import MultiEmbedding


def test_embeddings():
    embedding_sizes = {"a": (2, 2), "b": (3, 3)}
    total_embedding_size = sum([dim for (_, dim) in embedding_sizes.values()])
    data = {"a": torch.Tensor([[[1.0], [0.0], [1.0]]]), "b": torch.Tensor([[[1.0], [0.0], [2.0]]])}
    batch_size, time_len = data["a"].shape[0], data["a"].shape[1]
    embeddings = MultiEmbedding(embedding_sizes)
    output = embeddings(data)
    for key in embedding_sizes:
        n, dim = embedding_sizes[key]
        assert embeddings.embedding[key].weight.shape == (n + 1, dim)

    assert output.shape == (batch_size, time_len, total_embedding_size)
