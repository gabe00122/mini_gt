import flax.linen as nn

n = 6
scale = (9 * n) ** -(1 / 4)


class TransformerLayer(nn.Module):
    num_heads: int = 8
    token_features: int = 16

    @nn.compact
    def __call__(self, inputs, mask=None):
        x = inputs
        res = x

        # x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.token_features,
            kernel_init=nn.initializers.variance_scaling(
                scale, "fan_in", "truncated_normal"
            ),
        )(x, mask=mask)

        x += res
        res = x

        # x = nn.LayerNorm()(x)
        x = nn.Dense(
            features=self.token_features,
            kernel_init=nn.initializers.variance_scaling(
                scale, "fan_in", "truncated_normal"
            ),
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            features=self.token_features,
            kernel_init=nn.initializers.variance_scaling(
                scale, "fan_in", "truncated_normal"
            ),
        )(x)

        x += res

        return x


class Transformer(nn.Module):
    num_heads: int = 8
    token_features: int = 16
    vocab_size: int = 10
    num_layers: int = 6

    @nn.compact
    def __call__(self, inputs, mask=None):
        x = inputs
        for _ in range(self.num_layers):
            x = TransformerLayer(
                num_heads=self.num_heads,
                token_features=self.token_features,
            )(x, mask)

        # x = nn.LayerNorm()(x)
        x = nn.DenseGeneral(
            features=self.vocab_size,
            axis=(0, 1),
            kernel_init=nn.initializers.glorot_normal(),
        )(x)

        return x
