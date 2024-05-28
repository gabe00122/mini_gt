from typing import NamedTuple, Any
from functools import partial

import jax
from jax import random, Array, numpy as jnp
import optax
from optax.losses import softmax_cross_entropy
import numpy as np

from sequence_gym.env import create_training_batch, create_training_sample, TrainingSample
from sequence_gym.network import Network
from sequence_gym.positional_embeddings import get_positional_embeddings
from sequence_gym.transformer import Transformer
from sequence_gym.vocab import VocabDescribe


def main():
    rng_key = random.PRNGKey(42)
    vocab = VocabDescribe(10)
    embedding_features = 32
    sequence_length = 51
    num_heads = 8
    batch_size = 128

    transformer = Transformer(num_heads=num_heads, token_features=embedding_features)

    network = Network(
        transformer=transformer,
        seq_length=sequence_length,
        embedding_features=embedding_features,
        position_embeddings=get_positional_embeddings(sequence_length, embedding_features)
    )

    param_key, dummy_key, rng_key = random.split(rng_key, 3)
    dummy_batch = create_training_sample(dummy_key, vocab, sequence_length)

    network_params = network.init(param_key, dummy_batch.sequence)

    optimizer = optax.adam(
        learning_rate=optax.warmup_cosine_decay_schedule(0.000001, 0.01, 1_000, 9_000)
    )
    opt_state = optimizer.init(network_params)

    static_state = StaticState(network, optimizer, batch_size, sequence_length, vocab)
    training_state = TrainingState(rng_key, network_params, opt_state)

    total_steps = 10_000
    losses = np.zeros((total_steps,), dtype=np.float32)

    for i in range(total_steps):
        training_state, metrics = training_step(static_state, training_state)
        loss_value = metrics.loss.item()
        losses[i] = loss_value

        if i % 100 == 99:
            print(f"{i}: {loss_value}")

    np.save("losses2", losses)


def loss(network: Network, params, training_batch: TrainingSample):
    vec_network = jax.vmap(network.apply, in_axes=(None, 0, 0))

    logits = vec_network(params, training_batch.sequence, training_batch.mask)
    return jnp.mean(softmax_cross_entropy(logits, training_batch.label))


class StaticState(NamedTuple):
    network: Network
    solver: Any
    batch_size: int
    seq_length: int
    vocab: VocabDescribe


class TrainingState(NamedTuple):
    rng_key: Array
    params: Any
    opt_state: Any


class Metrics(NamedTuple):
    loss: Array


@partial(jax.jit, static_argnums=0)
def training_step(static_state: StaticState, state: TrainingState) -> tuple[TrainingState, Metrics]:
    rng_key = state.rng_key
    keys = random.split(rng_key, static_state.batch_size + 1)
    rng_key = keys[0]
    sample_keys = keys[1:]

    sample = create_training_batch(sample_keys, static_state.vocab, static_state.seq_length)

    loss_value, grad = jax.value_and_grad(loss, argnums=1)(static_state.network, state.params, sample)
    updates, opt_state = static_state.solver.update(grad, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)

    state = TrainingState(
        rng_key=rng_key,
        params=params,
        opt_state=opt_state,
    )
    metrics = Metrics(
        loss=loss_value,
    )

    return state, metrics


if __name__ == '__main__':
    main()
