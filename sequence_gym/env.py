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
    # position_array = jnp.full_like(indices, position, dtype=jnp.int32)

    mask = indices < position
    return mask


def create_training_sample(rng_key: ArrayLike, vocab: VocabDescribe, batch_length: int) -> TrainingSample:
    sequence_rng, length_rng, position_rng, roll_rng = random.split(rng_key, 4)
    length = random.randint(length_rng, (), 1, batch_length // 2)

    sequence = create_sequence_var(sequence_rng, vocab, length, batch_length)
    train_point = random.randint(position_rng, (), length + 1, length * 2)

    roll_amount = random.randint(roll_rng, (), 1, batch_length - train_point)
    sequence = jnp.roll(sequence, roll_amount)
    sequence = sequence.at[roll_amount - 1].set(vocab.reverse_token)

    train_point += roll_amount

    mask = create_mask(batch_length, train_point)
    return TrainingSample(
        sequence=sequence,
        mask=mask,
        label=nn.one_hot(sequence[train_point], vocab.total_tokens))


create_training_batch = jax.vmap(create_training_sample, in_axes=(0, None, None))


def create_sequence_var(rng_key: ArrayLike, vocab: VocabDescribe, sequence_length: Array, buffer_size: int):
    seq = generate_sequence(rng_key, vocab, buffer_size)
    reverse_seq = reverse_sequence(seq)
    reverse_seq = jnp.roll(reverse_seq, (sequence_length * 2) - buffer_size + 1)

    indices = jnp.arange(0, buffer_size)
    output = jnp.where(indices < sequence_length, seq, reverse_seq)
    output = output.at[sequence_length].set(vocab.reverse_token)

    # optional
    output = jnp.where(indices >= sequence_length * 2 + 1, -1, output)

    return output
