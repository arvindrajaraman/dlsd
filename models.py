from typing import List

from icecream import ic
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax

class QNetClassic(nn.Module):
    action_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.action_size)(x)
        return x

class QNetClassicCraftax(nn.Module):
    action_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, state):
        maps, metadata = jnp.split(state, [7 * 9 * 21], axis=-1)
        maps = maps.reshape((-1, 7, 9, 21))
        
        maps = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(maps)
        maps = nn.relu(maps)
        maps = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(maps)
        maps = nn.relu(maps)
        maps = maps.reshape((maps.shape[0], -1))

        y = jnp.concatenate((maps, metadata), axis=-1)
        y = nn.Dense(self.hidden_size)(y)
        y = nn.relu(y)
        y = nn.Dense(self.hidden_size)(y)
        y = nn.relu(y)
        y = nn.Dense(self.action_size)(y)
        return y

class QNet(nn.Module):
    action_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, state, skill):
        x = jnp.concatenate((state, skill), axis=-1)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.action_size)(x)
        return x

class QNetCraftax(nn.Module):
    action_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, state, skill):
        maps, metadata = jnp.split(state, [7 * 9 * 21], axis=-1)
        maps = maps.reshape((-1, 7, 9, 21))
        
        maps = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(maps)
        maps = nn.relu(maps)
        maps = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(maps)
        maps = nn.relu(maps)
        maps = maps.reshape((maps.shape[0], -1))

        y = jnp.concatenate((maps, metadata, skill), axis=-1)
        y = nn.Dense(self.hidden_size)(y)
        y = nn.relu(y)
        y = nn.Dense(self.hidden_size)(y)
        y = nn.relu(y)
        y = nn.Dense(self.action_size)(y)
        return y   
    
class QNetCraftaxAugmented(nn.Module):
    action_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, state, embedding, skill):
        maps, metadata = jnp.split(state, [7 * 9 * 21], axis=-1)
        maps = maps.reshape((-1, 7, 9, 21))
        
        maps = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(maps)
        maps = nn.relu(maps)
        maps = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(maps)
        maps = nn.relu(maps)
        maps = maps.reshape((maps.shape[0], -1))

        y = jnp.concatenate((maps, metadata, embedding, skill), axis=-1)
        y = nn.Dense(self.hidden_size)(y)
        y = nn.relu(y)
        y = nn.Dense(self.hidden_size)(y)
        y = nn.relu(y)
        y = nn.Dense(self.action_size)(y)
        return y    

class Discriminator(nn.Module):
    skill_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, state):
        x = nn.Dense(features=self.hidden_size)(state)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.skill_size)(x)
        return x

class DiscriminatorCraftax(nn.Module):
    skill_size: int
    hidden_size: int

    @nn.compact
    def __call__(self, state):
        maps, metadata = jnp.split(state, [7 * 9 * 21], axis=-1)
        maps = maps.reshape((-1, 7, 9, 21))
        
        maps = nn.Conv(features=32, kernel_size=(3, 3), padding='SAME')(maps)
        maps = nn.relu(maps)
        maps = nn.max_pool(maps, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        maps = nn.Conv(features=64, kernel_size=(3, 3), padding='SAME')(maps)
        maps = nn.relu(maps)
        maps = nn.max_pool(maps, window_shape=(2, 2), strides=(2, 2), padding='VALID')
        maps = maps.reshape((maps.shape[0], -1))
        
        x = jnp.concatenate((maps, metadata), axis=-1)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.skill_size)(x)
        return x

class APTForwardDynamicsModel(nn.Module):
    action_size: int
    skill_size: int
    hidden_size: int
    
    @nn.compact
    def __call__(self, embedding, action):
        action_1h = jax.nn.one_hot(action, self.action_size)
        x = jnp.concatenate((embedding, action_1h), axis=-1)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.skill_size)(x)
        return x
    
class APTBackwardDynamicsModel(nn.Module):
    action_size: int
    hidden_size: int
    
    @nn.compact
    def __call__(self, embedding1, embedding2):
        x = jnp.concatenate((embedding1, embedding2), axis=-1)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.action_size)(x)
        return x
