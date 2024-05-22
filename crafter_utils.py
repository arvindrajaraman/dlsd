import jax
import jax.numpy as jnp
import numpy as onp
import chex
from icecream import ic
from sentence_transformers import SentenceTransformer
from functools import partial
from typing import Optional, Tuple, Union, Any

from craftax.craftax_classic.constants import *
from gymnax.environments import environment, spaces
from sklearn.manifold import TSNE

from crafter_constants import blocks_labels, mobs_labels, inventory_labels

def crafter_score(achievement_counts, episodes):
    achievement_success_rates = ((achievement_counts / episodes) * 100.0)
    return onp.nan_to_num(
        onp.exp(onp.log(achievement_success_rates + 1.0).mean()) - 1.0,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

def separate_features(features, idxs):
    i = jnp.argsort(idxs)
    sorted_features = features[i]
    sorted_idxs = idxs[i]

    changes = jnp.where(jnp.diff(sorted_idxs) != 0)[0] + 1
    grouped_features = jnp.split(sorted_features, changes)
    grouped_idxs = jnp.split(sorted_idxs, changes)
    return grouped_features, grouped_idxs

def obs_to_features(obs):
    maps, metadata = jnp.split(obs, [7 * 9 * 21], axis=1)
    
    maps = jnp.reshape(maps, [-1, 7, 9, 21])
    maps = jnp.transpose(maps, [0, 3, 1, 2])
    maps = jnp.reshape(maps, [-1, 21, 7 * 9])
    maps = maps.sum(axis=2)

    features = jnp.concatenate((maps, metadata), axis=-1)
    return features

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)
    
class ParallelizedBatchEnvWrapper(GymnaxWrapper):
    """Batches reset and step functions"""

    def __init__(self, env: environment.Environment, num_envs: int):
        super().__init__(env)

        self.num_envs = num_envs

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(
        self, rng, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, state, reward, done, info = self.step_fn(rngs, state, action, params)

        return obs, state, reward, done, info

# OBS_SIZE = 1345
# MAP_HEIGHT = 7
# MAP_WIDTH = 9
# MAP_FEATURES = 21

# obs_feature_labels = {
#     -22: 'inv/wood',
#     -21: 'inv/stone',
#     -20: 'inv/coal',
#     -19: 'inv/iron',
#     -18: 'inv/diamond',
#     -17: 'inv/sapling',
#     -16: 'inv/wood_pickaxe',
#     -15: 'inv/stone_pickaxe',
#     -14: 'inv/iron_pickaxe',
#     -13: 'inv/wood_sword',
#     -12: 'inv/stone_sword',
#     -11: 'inv/iron_sword',
#     -10: 'int/player_health',
#     -9: 'int/player_food',
#     -8: 'int/player_drink',
#     -7: 'int/player_energy',
#     -6: 'dir/1',
#     -5: 'dir/2',
#     -4: 'dir/3',
#     -3: 'dir/4',
#     -2: 'light_level',
#     -1: 'is_sleeping',
# }

# map_feature_idxs = {
#     0: 'block/invalid',
#     1: 'block/out_of_bounds',
#     2: 'block/grass',
#     3: 'block/water',
#     4: 'block/stone',
#     5: 'block/tree',
#     6: 'block/wood',
#     7: 'block/path',
#     8: 'block/coal',
#     9: 'block/iron',
#     10: 'block/diamond',
#     11: 'block/crafting_table',
#     12: 'block/furnace',
#     13: 'block/sand',
#     14: 'block/lava',
#     15: 'block/plant',
#     16: 'block/ripe_plant',
#     17: 'mob/zombie',
#     18: 'mob/cow',
#     19: 'mob/skeleton',
#     20: 'mob/arrow'
# }

# # Replace negative indices with positive indices
# for idx in obs_feature_labels:
#     if idx < 0:
#         obs_feature_labels[idx + OBS_SIZE] = obs_feature_labels[idx]

# for h in range(MAP_HEIGHT):
#     for w in range(MAP_WIDTH):
#         for f in range(MAP_FEATURES):
#             map_feature_name = map_feature_idxs[f]
#             idx = (h * MAP_WIDTH * MAP_FEATURES) + (w * MAP_FEATURES) + f
#             obs_feature_labels[idx] = f'{map_feature_name}/h{h}-w{w}'

def new_embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

def embedding_crafter(embedding_model, next_obs):
    batch_size = next_obs.shape[0]
    maps, metadata = jnp.split(next_obs, [7 * 9 * 21], axis=1)
    
    # For each block/mob type, count how many are in frame
    maps = jnp.reshape(maps, [-1, 7, 9, 21])
    maps = jnp.transpose(maps, [0, 3, 1, 2])
    maps = jnp.reshape(maps, [-1, 21, 7 * 9])
    maps = maps.sum(axis=2)
    maps = jnp.round(maps).astype(jnp.int32)
    blocks, mobs = jnp.split(maps, [17], axis=1)

    # Extract and format metadata
    inventory, intrinsics, direction, light_level, is_sleeping = jnp.split(metadata, [12, 16, 20, 21], axis=1)
    inventory = jnp.round(inventory * 10.0).astype(jnp.int32)
    intrinsics *= 10.0

    health, food, drink, energy = jnp.split(intrinsics, [1, 2, 3], axis=1)

    # Construct embedding
    sentences = []
    for i in range(batch_size):
        blocks_gt0 = jnp.argwhere(blocks[i] > 0).reshape(-1)
        mobs_gt0 = jnp.argwhere(mobs[i] > 0).reshape(-1)
        inventory_gt0 = jnp.argwhere(inventory[i] > 0).reshape(-1)

        blocks_str = 'You see {}.'.format(', '.join([blocks_labels[b] for b in blocks_gt0]))
        mobs_str = 'You see {}.'.format(', '.join([mobs_labels[m] for m in mobs_gt0]))
        inventory_str = 'You have in your inventory {}.'.format(', '.join([inventory_labels[i] for i in inventory_gt0]))

        status = []
        if food[i].item() < 10.0 - 1e-4:
            status.append("hungry")
        if drink[i].item() < 10.0 - 1e-4:
            status.append("thirsty")
        if energy[i].item() < 10.0 - 1e-4:
            status.append("tired")
        status_str = 'You feel {}.'.format(', '.join(status))

        if health[i].item() < 5.0 - 1e-4:
            health_str = 'You are at low health.'
        elif health[i].item() < 10.0 - 1e-4:
            health_str = 'You are at moderate health.'
        else:
            health_str = 'You are at full health.'

        desc = [status_str, health_str]
        if len(blocks_gt0) > 0:
            desc.append(blocks_str)
        if len(mobs_gt0) > 0:
            desc.append(mobs_str)
        if len(inventory_gt0) > 0:
            desc.append(inventory_str)

        desc = ' '.join(desc)
        sentences.append(desc)

    embeddings = embedding_model.encode(sentences)
    embeddings = jnp.array(embeddings)
    return embeddings, sentences

def tsne_embeddings(embeddings):
    print('Running t-SNE')
    return TSNE(verbose=1).fit_transform(embeddings)

def bin_timesteps(timesteps):
    bins = jnp.empty_like(timesteps, dtype=jnp.int32)
    bins = bins.at[(timesteps < 50)].set(0)
    bins = bins.at[(timesteps >= 50) & (timesteps < 100)].set(1)
    bins = bins.at[(timesteps >= 100) & (timesteps < 150)].set(2)
    bins = bins.at[(timesteps >= 150) & (timesteps < 200)].set(3)
    bins = bins.at[(timesteps >= 200) & (timesteps < 250)].set(4)
    bins = bins.at[(timesteps >= 250) & (timesteps < 300)].set(5)
    bins = bins.at[(timesteps >= 300) & (timesteps < 400)].set(6)
    bins = bins.at[(timesteps >= 400)].set(7)
    return bins

from craftax.craftax_classic.constants import BlockType, DIRECTIONS, Action
from functools import partial
from jax import jit

@partial(jit, static_argnames=('vectorization',))
def is_mining_sapling(key, state, action, vectorization):
    key, subkey = jax.random.split(key)
    block_position = state.player_position + DIRECTIONS[state.player_direction]
    is_mining_sapling = jnp.logical_and(
        state.map[jnp.arange(vectorization), block_position[:, 0], block_position[:, 1]] == BlockType.GRASS.value,
        jax.random.uniform(subkey) < 0.1,
    )
    doing_do = (action == Action.DO)
    return key, jnp.logical_and(is_mining_sapling, doing_do)
