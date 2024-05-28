import flax.linen as nn


class TransformerLayer(nn.Module):
    num_heads: int = 8
    token_features: int = 16

    @nn.compact
    def __call__(self, inputs, mask=None):
        x = inputs
        res = x

        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.token_features
        )(x, mask=mask)

        x = nn.LayerNorm()(x + res)
        res = x

        x = nn.Dense(features=self.token_features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.token_features)(x)

        x = nn.LayerNorm()(x + res)

        return x


class Transformer(nn.Module):
    num_heads: int = 8
    token_features: int = 16
    vocab_size: int = 10
    num_layers: int = 3

    @nn.compact
    def __call__(self, inputs, mask=None):
        x = inputs
        for _ in range(self.num_layers):
            x = TransformerLayer(
                num_heads=self.num_heads,
                token_features=self.token_features,
            )(x, mask)

        x = nn.DenseGeneral(features=self.vocab_size, axis=(0, 1))(x)

        return x
