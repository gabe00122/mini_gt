import jax
from jax import random, numpy as jnp, Array
from jax.typing import ArrayLike
from sequence_gym.vocab import VocabDescribe
from typing import NamedTuple
from flax import linen as nn


def generate_sequence(rng_key: ArrayLike, vocab: VocabDescribe, sequence_length: int) -> Array:
    return random.randint(rng_key, (sequence_length,), vocab.special_tokens, vocab.total_tokens)


def reverse_sequence(seq: ArrayLike) -> Array:
    return jnp.flip(seq, 0)


def create_sequence(rng_key: ArrayLike, vocab: VocabDescribe, sequence_length: int) -> Array:
    seq = generate_sequence(rng_key, vocab, (sequence_length - 1) // 2)
    reverse_seq = reverse_sequence(seq)

    return jnp.hstack([seq, jnp.array([vocab.reverse_token]), reverse_seq])


class TrainingSample(NamedTuple):
    sequence: Array
    mask: Array
    label: Array


def create_mask(length: int, position: Array):
    indices = jnp.arange(0, length, dtype=jnp.int32)
    position_array = jnp.full_like(indices, position, dtype=jnp.int32)

    mask = indices < position_array
    return mask


def create_training_sample(rng_key: ArrayLike, vocab: VocabDescribe, sequence_length: int) -> TrainingSample:
    sequence_rng, position_rng = random.split(rng_key)

    position = random.randint(position_rng, (), sequence_length // 2 + 1, sequence_length)
    sequence = create_sequence(sequence_rng, vocab, sequence_length)
    mask = create_mask(sequence_length, position)
    return TrainingSample(
        sequence=sequence,
        mask=mask,
        label=nn.one_hot(sequence[position], vocab.total_tokens))


create_training_batch = jax.vmap(create_training_sample, in_axes=(0, None, None))
