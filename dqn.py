
# python
import argparse
from collections import deque
from datetime import datetime
import math
import os
from pathlib import Path
import time
import yaml

# packages
from craftax.craftax_env import make_craftax_env_from_name
from dotenv import load_dotenv
import flashbax as fbx
from flax import linen as nn
from functools import partial
import gym
from icecream import ic
from jax import random, jit, vmap, pmap
import jax
import jax.numpy as jnp
import jax.lax as lax
from ml_collections import ConfigDict
import numpy as onp
import optax
from tqdm import tqdm
import wandb

# files
from models import QNetClassic, QNetClassicCraftax
from structures import ReplayBuffer
from utils import grad_norm
import lunar_vis
import crafter_constants
import crafter_utils
import crafter_vis

def train(key, config, run_name, log):
    # region 1. Initialize training state
    key, qlocal_key, qtarget_key, env_key = jax.random.split(key, num=4)

    QNetModel = QNetClassicCraftax if config.env_name == 'Craftax-Classic-Symbolic-v1' else QNetClassic
    dummy_state = jnp.zeros((1, config.state_size))

    qlocal = QNetModel(
        action_size=config.action_size,
        hidden1_size=config.policy_units,
        hidden2_size=config.policy_units
    )
    qlocal_params = qlocal.init(qlocal_key, dummy_state)
    qlocal_opt = optax.adam(learning_rate=config.policy_lr)
    qlocal_opt_state = qlocal_opt.init(qlocal_params)

    qtarget = QNetModel(
        action_size=config.action_size,
        hidden1_size=config.policy_units,
        hidden2_size=config.policy_units
    )
    qtarget_params = qtarget.init(qtarget_key, dummy_state)

    buffer = ReplayBuffer.create({
        'obs': jnp.zeros((config.state_size,)),
        'action': jnp.array(0, dtype=jnp.int32),
        'reward': jnp.array(0.0),
        'next_obs': jnp.zeros((config.state_size,)),
        'done': jnp.array(False, dtype=jnp.bool_),
    }, size=config.buffer_size)

    if config.env_name == 'LunarLander-v2':
        min_x = -1.
        max_x = 1.
        min_y = 0.
        max_y = 1.
        num_bins_x = num_bins_y = 50
        bin_size_x = (max_x - min_x) / num_bins_x
        bin_size_y = (max_y - min_y) / num_bins_y
        visits = onp.zeros((num_bins_x, num_bins_y))
    elif config.env_name == 'Craftax-Classic-Symbolic-v1':
        achievement_counts_total = onp.zeros((22,), dtype=jnp.int32)
    # endregion

    # 2. Define model usage/update functions
    @jit
    def act(key, qlocal_params, obs, eps=0.0):
        key, explore_key, action_key = random.split(key, num=3)
        action = jnp.where(
            random.uniform(explore_key, shape=(config.num_workers,)) > eps,
            jnp.argmax(qlocal.apply(qlocal_params, obs), axis=1),
            random.choice(action_key, jnp.arange(config.action_size), shape=(config.num_workers,))
        )
        return key, action

    @jit
    def update_q(qlocal_params, qlocal_opt_state, qtarget_params, batch):
        def loss_fn(qlocal_params, batch):
            obs, actions, rewards, next_obs, dones = batch['obs'], batch['action'], batch['reward'], batch['next_obs'], batch['done']

            Q_targets_next = qtarget.apply(qtarget_params, next_obs).max(axis=1)
            Q_targets = rewards + (config.gamma * Q_targets_next * ~dones)
            Q_expected = qlocal.apply(qlocal_params, obs)[jnp.arange(config.batch_size), actions]

            loss = optax.squared_error(Q_targets, Q_expected).mean()

            q_value_mean = Q_expected.mean()
            target_value_mean = Q_targets.mean()
            reward_mean = rewards.mean()
            return loss, (q_value_mean, target_value_mean, reward_mean)

        (loss, aux_metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(qlocal_params, batch)
        updates, qlocal_opt_state = qlocal_opt.update(grads, qlocal_opt_state)
        qlocal_params = optax.apply_updates(qlocal_params, updates)

        q_value_mean, target_value_mean, reward_mean = aux_metrics
        qlocal_metrics = {
            'critic_loss': loss,
            'q_values': q_value_mean,
            'target_values': target_value_mean,
            'qlocal_grad_norm': grad_norm(grads),
            'reward': reward_mean
        }

        # Update target network
        qtarget_params = optax.incremental_update(qlocal_params, qtarget_params, config.tau)

        return qlocal_params, qlocal_opt_state, qtarget_params, qlocal_metrics

    # 2. Run training
    if config.gym_np:
        env = gym.vector.make(config.env_name, num_envs=config.num_workers)
        obs = jnp.array(env.reset())
        env_reset = env.reset
        env_step = env.step
        state, next_state = None, None
    else:
        env = make_craftax_env_from_name(config.env_name, auto_reset=True)
        env_params = env.default_params
        env_reset = pmap(env.reset, in_axes=(0, None))
        env_step = pmap(env.step, in_axes=(0, 0, 0, None))
        obs, state = env_reset(random.split(env_key, config.num_workers), env_params)

    # Divide frequencies/steps by num_workers
    config.steps = math.ceil(config.steps / config.num_workers)
    config.warmup_steps = math.ceil(config.warmup_steps / config.num_workers)
    config.save_freq = math.ceil(config.save_freq / config.num_workers)
    config.vis_freq = math.ceil(config.vis_freq / config.num_workers)
    config.model_update_freq = math.ceil(config.model_update_freq / config.num_workers)

    score_gt = jnp.zeros((config.num_workers,))
    steps_per_ep = jnp.zeros((config.num_workers,), dtype=jnp.int32)

    eps = config.eps_start
    score_gt_window = deque(maxlen=100)
    episodes = 0
    metrics = dict()
    for t_step in tqdm(range(config.steps), unit_scale=config.num_workers):
        key, action = act(key, qlocal_params, obs, eps)
        if config.gym_np:
            action = onp.asarray(action)
            next_obs, reward_gt, done, _ = env_step(action)
            next_obs, reward_gt, done = jnp.array(next_obs), jnp.array(reward_gt), jnp.array(done)
        else:
            key, step_key = random.split(key)
            next_obs, next_state, reward_gt, done, _ = env_step(random.split(step_key, config.num_workers), state, action, env_params)

        done_idx = jnp.argwhere(
            jnp.logical_or(done, steps_per_ep >= config.max_steps_per_ep)
        )
    
        steps_per_ep += 1
        score_gt += reward_gt
        for idx in done_idx:
            episodes += 1
            score_gt_window.append(score_gt[idx])
            metrics['score/gt'] = jnp.mean(jnp.array(score_gt_window))
            metrics['steps_per_ep'] = jnp.mean(steps_per_ep[done_idx])

        if t_step % config.log_freq == 0:
            metrics['eps'] = eps
            metrics['episodes'] = episodes

        buffer.add_transition_batch({
            'obs': obs,
            'action': action,
            'reward': reward_gt,
            'next_obs': next_obs,
            'done': done,
        }, batch_size=config.num_workers)

        if t_step > config.warmup_steps:
            batch = buffer.sample(config.batch_size)
            if t_step % config.model_update_freq == 0:
                qlocal_params, qlocal_opt_state, qtarget_params, qlocal_metrics = update_q(qlocal_params, qlocal_opt_state, qtarget_params, batch)
                metrics.update(qlocal_metrics)

            if config.env_name == 'LunarLander-v2':
                x_index = ((obs[:, 0] - min_x) / bin_size_x).astype(int)
                x_index = onp.minimum(onp.maximum(x_index, 0), num_bins_x - 1)
                y_index = ((obs[:, 1] - min_y) / bin_size_y).astype(int)
                y_index = onp.minimum(onp.maximum(y_index, 0), num_bins_y - 1)

                visits[y_index, x_index] += 1

                if t_step % config.vis_freq == 0:
                    metrics["log_visits_all"] = lunar_vis.plot_visits(visits, min_x, max_x, min_y, max_y, log=True)
                    visits = onp.zeros_like(visits)

                if (t_step % config.vis_freq == 0 or t_step == config.steps - 1) and config.save_rollouts:
                    rollouts = lunar_vis.gather_rollouts(qlocal, qlocal_params, config.rollouts_per_skill, config.max_steps_per_ep)
                    for rollout_num in range(config.rollouts_per_skill):
                        rollout_filepath = rollouts[rollout_num]
                        metrics[f'rollouts/num_{rollout_num}'] = wandb.Video(rollout_filepath, fps=16, format="mp4")
            elif config.env_name == 'Craftax-Classic-Symbolic-v1':
                # Achievement counts
                for idx in done_idx:
                    achievement_counts_total += state.achievements[idx]

                if t_step % config.log_freq == 0:
                    for idx, label in enumerate(crafter_constants.achievement_labels):
                        metrics[f'achievements/{label}'] = achievement_counts_total[idx] / episodes
                    metrics['score/crafter'] = crafter_utils.crafter_score(achievement_counts_total, episodes)

                if (t_step % config.vis_freq == 0 or t_step == config.steps - 1) and config.save_rollouts:
                    rollouts = crafter_vis.gather_rollouts(key, qlocal, qlocal_params, config.rollouts_per_skill, config.max_steps_per_ep)
                    for rollout_num in range(config.rollouts_per_skill):
                        rollout_filepath = rollouts[rollout_num]
                        metrics[f'rollouts/num_{rollout_num}'] = wandb.Video(rollout_filepath, fps=8, format="mp4")
        
        if log:
            wandb.log(metrics, step=t_step*config.num_workers)
            metrics.clear()

        if log and config.save_checkpoints and (t_step % config.save_freq == 0 or t_step == config.steps - 1):
            jnp.savez(os.path.join(os.getcwd(), 'data', run_name),
                t_step=t_step,
                qlocal_params=qlocal_params,
                qtarget_params=qtarget_params)

        obs, state = next_obs, next_state
        if t_step > config.warmup_steps:
            eps = max(eps * (config.eps_decay ** config.num_workers), config.eps_end)
        score_gt = score_gt.at[done_idx].set(0.0)
        steps_per_ep = steps_per_ep.at[done_idx].set(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--nolog', action='store_true')
    args = parser.parse_args()

    config = yaml.safe_load(Path(os.path.join('./config', args.config)).read_text())
    if "seed" not in config:
        config['seed'] = int(datetime.now().timestamp())
    config = ConfigDict(config)

    if not args.nolog:
        run_name = '{}_dqn_{}'.format(config.env_name, int(datetime.now().timestamp()))
        os.makedirs(f'./data/{run_name}')

        load_dotenv()
        wandb.login(key=os.getenv("WANDB_API_KEY"))

        run = wandb.init(
            project="language-skills",
            entity=os.getenv("WANDB_ENTITY"),
            name=run_name,
            config=config
        )
        wandb.config = config
    else:
        run_name = None

    devices = jax.devices()
    print(f'{len(devices)} devices visible through JAX: {devices}')
    config.num_workers = len(devices)

    key = random.PRNGKey(config.seed)
    train(key, config, run_name, log=not args.nolog)
    run.finish()
