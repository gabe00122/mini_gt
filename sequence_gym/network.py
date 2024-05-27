from flax import linen as nn
from jax import Array

from .transformer import Transformer


class Network(nn.Module):
    seq_length: int
    embedding_features: int
    transformer: Transformer
    position_embeddings: Array

    @nn.compact
    def __call__(self, inputs, mask=None):
        token_embeddings = nn.Embed(
            num_embeddings=self.seq_length,
            features=self.embedding_features,
        )(inputs)
        embeddings = token_embeddings + self.position_embeddings
        return self.transformer(embeddings, mask)

    def __hash__(self):
        return id(self)
