#region Imports
import argparse
from collections import deque
from datetime import datetime
import math
import os
from pathlib import Path
import yaml

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

from models import QNet, QNetCraftax, Discriminator, DiscriminatorCraftax
from structures import ReplayBuffer
from utils import grad_norm
import crafter_constants
import crafter_utils
import crafter_vis
import lunar_utils
import lunar_vis
#endregion

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(key, config, run_name, log):
    #region 1. Initialize training state
    key, qlocal_key, qtarget_key, discrim_key, env_key, skill_init_key = jax.random.split(key, num=6)

    QNetModel = QNetCraftax if config.env_name == 'Craftax-Classic-Symbolic-v1' else QNet
    DiscriminatorModel = DiscriminatorCraftax if (config.env_name == 'Craftax-Classic-Symbolic-v1' and config.embedding_type == 'identity') else Discriminator

    dummy_state = jnp.zeros((1, config.state_size))
    dummy_embedding = jnp.zeros((1, config.embedding_size))
    dummy_skill = jnp.zeros((1, config.skill_size))

    qlocal = QNetModel(
        action_size=config.action_size,
        hidden_size=config.policy_units
    )
    qlocal_params = qlocal.init(qlocal_key, dummy_state, dummy_skill)
    qlocal_opt = optax.adam(learning_rate=config.policy_lr)
    qlocal_opt_state = qlocal_opt.init(qlocal_params)

    qtarget = QNetModel(
        action_size=config.action_size,
        hidden_size=config.policy_units
    )
    qtarget_params = qtarget.init(qtarget_key, dummy_state, dummy_skill)

    discrim = DiscriminatorModel(
        skill_size=config.skill_size,
        hidden_size=config.discrim_units
    )
    discrim_params = discrim.init(discrim_key, dummy_embedding)
    discrim_opt = optax.adam(learning_rate=config.discrim_lr)
    discrim_opt_state = discrim_opt.init(discrim_params)

    # # Set discrim params to perfect discriminator (for debugging)
    # discrim_params = checkpoints.restore_checkpoint(os.path.join(os.getcwd(), 'data', 'perfect_discrim'), target=None)['discrim_params']

    buffer = ReplayBuffer.create({
        'obs': jnp.zeros((config.state_size,)),
        'action': jnp.array(0, dtype=jnp.int32),
        'skill': jnp.zeros((config.skill_size,)),
        'next_obs': jnp.zeros((config.state_size,)),
        'next_obs_embedding': jnp.zeros((config.embedding_size,)),
        'done': jnp.array(False, dtype=jnp.bool_),
    }, size=config.buffer_size)

    if config.embedding_type == 'identity':
        embedding_fn = lambda next_obs: (next_obs, None)
    elif config.embedding_type == '1h':
        embedding_fn = lambda next_obs: lunar_utils.embedding_1he(next_obs[:, 0])
    elif config.embedding_type == 'crafter':
        embedding_model = crafter_utils.new_embedding_model()
        embedding_fn = lambda next_obs: crafter_utils.embedding_crafter(embedding_model, next_obs)
    else:
        raise ValueError(f"Invalid embedding type: {config['embedding_type']}")

    action_skill_mtx = onp.zeros((config.skill_size, config.action_size))
    if config.env_name == 'LunarLander-v2' and config.skill_size == 3 and config.embedding_type == '1h':
        skill_goal_mtx = onp.zeros((3, 3))
        min_x = -1.
        max_x = 1.
        min_y = 0.
        max_y = 1.
        num_bins_x = num_bins_y = 50
        bin_size_x = (max_x - min_x) / num_bins_x
        bin_size_y = (max_y - min_y) / num_bins_y
        visits = onp.zeros((config.skill_size + 1, num_bins_x, num_bins_y))
    elif config.env_name == 'Craftax-Classic-Symbolic-v1':
        achievement_counts = onp.zeros((config.skill_size, 22), dtype=onp.int32)  # refreshes every config.vis_step steps
        achievement_counts_total = onp.zeros((22,), dtype=onp.int32)  # does not refresh
        score_crafter_by_skill = []
        for _ in range(config.skill_size):
            score_crafter_by_skill.append(deque(maxlen=100))
    #endregion

    #region 2. Define model update functions
    @jit
    def act(key, qlocal_params, obs, skill, eps=0.0):
        key, explore_key, action_key = random.split(key, num=3)
        action = jnp.where(
            random.uniform(explore_key, shape=(config.num_workers,)) > eps,
            jnp.argmax(qlocal.apply(qlocal_params, obs, skill), axis=1),
            random.choice(action_key, jnp.arange(config.action_size), shape=(config.num_workers,))
        )
        return key, action

    @jit
    def update_model(qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, discrim_opt_state, batch):
        def q_loss_fn(qlocal_params, batch):
            obs, actions, skills, next_obs, next_obs_embeddings, dones = batch['obs'], batch['action'], batch['skill'], batch['next_obs'], batch['next_obs_embedding'], batch['done']

            skill_idxs = jnp.argmax(skills, axis=1)
            skill_preds = jax.nn.log_softmax(discrim.apply(discrim_params, next_obs_embeddings), axis=1)
            
            selected_logits = skill_preds[jnp.arange(config.batch_size), skill_idxs]
            rewards = (selected_logits + jnp.log(config.skill_size))

            Q_targets_next = qtarget.apply(qtarget_params, next_obs, skills).max(axis=1)
            Q_targets = rewards + (config.gamma * Q_targets_next * ~dones)
            Q_expected = qlocal.apply(qlocal_params, obs, skills)[jnp.arange(config.batch_size), actions]

            loss = optax.squared_error(Q_targets, Q_expected).mean()

            q_value_mean = Q_expected.mean()
            target_value_mean = Q_targets.mean()
            reward_mean = rewards.mean()
            return loss, (q_value_mean, target_value_mean, reward_mean)

        (q_loss, (q_value_mean, target_value_mean, reward_mean)), q_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(qlocal_params, batch)
        qlocal_updates, qlocal_opt_state = qlocal_opt.update(q_grads, qlocal_opt_state)
        qlocal_params = optax.apply_updates(qlocal_params, qlocal_updates)
        qtarget_params = optax.incremental_update(qlocal_params, qtarget_params, config.tau)

        def discrim_loss_fn(discrim_params, batch):
            next_obs_embeddings, skills = batch['next_obs_embedding'], batch['skill']

            skill_idxs = jnp.argmax(skills, axis=1)
            skill_preds = discrim.apply(discrim_params, next_obs_embeddings)
            skill_pred_idxs = jnp.argmax(skill_preds, axis=1)

            loss = optax.softmax_cross_entropy(skill_preds, skills).mean()
            acc = (skill_pred_idxs == skill_idxs).mean()
            pplx = 2. ** jax.scipy.special.entr(nn.activation.softmax(skill_preds)).sum(axis=1).mean()

            return loss, (acc, pplx)
        
        (discrim_loss, (discrim_acc, discrim_pplx)), discrim_grads = jax.value_and_grad(discrim_loss_fn, has_aux=True)(discrim_params, batch)
        discrim_updates, discrim_opt_state = discrim_opt.update(discrim_grads, discrim_opt_state)
        discrim_params = optax.apply_updates(discrim_params, discrim_updates)

        model_metrics = {
            'critic_loss': q_loss,
            'q_values': q_value_mean,
            'target_values': target_value_mean,
            'qlocal_grad_norm': grad_norm(q_grads),
            'reward': reward_mean,

            'discrim_loss': discrim_loss,
            'discrim_acc': discrim_acc,
            'discrim_pplx': discrim_pplx,
            'discrim_grad_norm': grad_norm(discrim_grads)
        }
        return qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, discrim_opt_state, model_metrics
    #endregion

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
        print(vars(state))
        raise NotImplementedError

    # Divide frequencies/steps by num_workers
    config.steps = math.ceil(config.steps / config.num_workers)
    config.warmup_steps = math.ceil(config.warmup_steps / config.num_workers)
    config.save_freq = math.ceil(config.save_freq / config.num_workers)
    config.vis_freq = math.ceil(config.vis_freq / config.num_workers)
    config.model_update_freq = math.ceil(config.model_update_freq / config.num_workers)

    score_gt = jnp.zeros((config.num_workers,))
    score_pr = jnp.zeros((config.num_workers,))
    steps_per_ep = jnp.zeros((config.num_workers,), dtype=jnp.int32)
    skill_idx = random.randint(skill_init_key, minval=0, maxval=config.skill_size, shape=(config.num_workers,))

    eps = config.eps_start
    score_gt_window = deque(maxlen=100)
    score_pr_window = deque(maxlen=100)
    episodes = 0

    score_gt_window_by_skill = []
    score_pr_window_by_skill = []
    for _ in range(config.skill_size):
        score_gt_window_by_skill.append(deque(maxlen=100))
        score_pr_window_by_skill.append(deque(maxlen=100))

    metrics = dict()
    for t_step in tqdm(range(config.steps), unit_scale=config.num_workers):
        skill = jax.nn.one_hot(skill_idx, config.skill_size)
        key, action = act(key, qlocal_params, obs, skill, eps)
        if config.gym_np:
            action = onp.asarray(action)
            next_obs, reward_gt, done, _ = env_step(action)
            next_obs, reward_gt, done = jnp.array(next_obs), jnp.array(reward_gt), jnp.array(done)
        else:
            key, step_key = random.split(key)
            next_obs, next_state, reward_gt, done, _ = env_step(random.split(step_key, config.num_workers), state, action, env_params)
        next_obs_embedding, _ = embedding_fn(next_obs)
        reward_pr = jax.nn.log_softmax(discrim.apply(discrim_params, next_obs_embedding), axis=1)[jnp.arange(config.num_workers), skill_idx] - jnp.log(1 / config.skill_size)
        done_idx = jnp.argwhere(
            jnp.logical_or(done, steps_per_ep >= config.max_steps_per_ep)
        )

        steps_per_ep += 1
        score_gt += reward_gt
        score_pr += reward_pr
        for idx in done_idx:
            skill_idx_done = skill_idx[idx].item()
            episodes += 1
            score_gt_window.append(score_gt)
            metrics['score/gt'] = jnp.mean(jnp.array(score_gt_window))
            score_gt_window_by_skill[skill_idx_done].append(score_gt)
            metrics[f'score/gt_s{skill_idx_done}'] = jnp.mean(jnp.array(score_gt_window_by_skill[skill_idx_done]))
            score_pr_window.append(score_pr)
            metrics['score/pr'] = jnp.mean(jnp.array(score_pr_window))
            score_pr_window_by_skill[skill_idx_done].append(score_pr)
            metrics[f'score/pr_s{skill_idx_done}'] = jnp.mean(jnp.array(score_pr_window_by_skill[skill_idx_done]))
            metrics[f'steps_per_ep'] = steps_per_ep[idx].item()
        metrics['steps'] = t_step * config.num_workers
        metrics['eps'] = eps
        metrics['episodes'] = episodes

        buffer.add_transition_batch({
            'obs': obs,
            'action': action,
            'skill': skill,
            'next_obs': next_obs,
            'next_obs_embedding': next_obs_embedding,
            'done': done
        }, batch_size=config.num_workers)

        if t_step > config.warmup_steps:
            batch = buffer.sample(config.batch_size)
            if t_step % config.model_update_freq == 0:
                qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, discrim_opt_state, model_metrics = update_model(qlocal_params, qlocal_opt_state, qtarget_params, discrim_params, discrim_opt_state, batch)
                metrics.update(model_metrics)

            action_skill_mtx[skill_idx, action] += 1
            if config.env_name == 'LunarLander-v2' and config.skill_size == 3 and config.embedding_type == '1h':
                # Visits
                x_index = ((obs[:, 0] - min_x) / bin_size_x).astype(int)
                x_index = onp.minimum(onp.maximum(x_index, 0), num_bins_x - 1)
                y_index = ((obs[:, 1] - min_y) / bin_size_y).astype(int)
                y_index = onp.minimum(onp.maximum(y_index, 0), num_bins_y - 1)
                visits[skill_idx, y_index, x_index] += 1
                visits[-1, y_index, x_index] += 1

                # Goal classification
                for idx in done_idx:
                    goal_idx = lunar_utils.classify_goal(obs[idx, 0])
                    skill_goal_mtx[skill_idx[idx].item(), goal_idx] += 1

                if (t_step % config.vis_freq == 10 or t_step == config.steps - 1):
                    for i in range(config.skill_size):
                        metrics[f"log_visits_s{i}"] = lunar_vis.plot_visits(visits[i], min_x, max_x, min_y, max_y, i, log=True)
                    
                    metrics.update({
                        "log_visits_all": lunar_vis.plot_visits(visits[-1], min_x, max_x, min_y, max_y, log=True),
                        "goal_confusion": lunar_vis.goal_skill_heatmap(skill_goal_mtx),
                        "action_confusion": lunar_vis.action_skill_heatmap(action_skill_mtx),
                        "I(z;s)": lunar_utils.mutual_information(skill_goal_mtx),
                        "I(z;a)": lunar_utils.mutual_information(action_skill_mtx),
                        "pred_landscape": lunar_vis.plot_pred_landscape(discrim, discrim_params, embedding_fn),
                    })

                    if config.save_rollouts:
                        print('Saving rollouts...')
                        raise NotImplementedError("need to fix gather_rollouts_by_skill impl for diayn.py")
                        rollouts = lunar_vis.gather_rollouts_by_skill(qlocal, qlocal_params, discrim, discrim_params, embedding_fn, config.skill_size, config.rollouts_per_skill, config.max_steps_per_ep)
                        for idx in range(config.skill_size):
                            for rollout_num in range(config.rollouts_per_skill):
                                rollout_filepath = rollouts[idx][rollout_num]
                                metrics[f'rollouts/skill_{idx}_num_{rollout_num}'] = wandb.Video(rollout_filepath, fps=16, format="mp4")

                    skill_goal_mtx = onp.zeros_like(skill_goal_mtx)
                    action_skill_mtx = onp.zeros_like(action_skill_mtx)
                    visits = onp.zeros_like(visits)
            elif config.env_name == 'Craftax-Classic-Symbolic-v1':
                # Achievement counts
                for idx in done_idx:
                    achievement_counts[skill_idx] += state.achievements[idx]
                    achievement_counts_total += state.achievements[idx]
        
                for idx, label in enumerate(crafter_constants.achievement_labels):
                    metrics[f'achievements/{label}'] = achievement_counts_total[idx] / episodes
                metrics['score/crafter'] = crafter_utils.crafter_score(achievement_counts_total, episodes)
                metrics[f'score/crafter_s{skill_idx}'] = crafter_utils.crafter_score(achievement_counts, episodes / config.skill_size)
                
                if (t_step % config.vis_freq == 10 or t_step == config.steps - 1):
                    metrics['achievement_counts'] = crafter_vis.achievement_counts_heatmap(achievement_counts, config.skill_size)
                    metrics['achievement_counts_log'] = crafter_vis.achievement_counts_heatmap(
                        jnp.where(achievement_counts > 0, jnp.log(achievement_counts), 0), config.skill_size)
                    metrics['action_confusion'] = crafter_vis.action_skill_heatmap(action_skill_mtx)
                    
                    if config.save_rollouts:
                        print('Saving rollouts...')
                        rollouts = crafter_vis.gather_rollouts_by_skill(key, qlocal, qlocal_params, discrim, discrim_params, embedding_fn, config.skill_size, config.rollouts_per_skill, config.max_steps_per_ep)
                        for idx in range(config.skill_size):
                            for rollout_num in range(config.rollouts_per_skill):
                                rollout_filepath = rollouts[idx][rollout_num]
                                metrics[f'rollouts/skill_{idx}_num_{rollout_num}'] = wandb.Video(rollout_filepath, fps=8, format="mp4")

                    # entire_rb = buffer.sample_all()
                    # all_skills, all_next_obs, all_next_obs_embeddings = entire_rb['skill'], entire_rb['next_obs'], entire_rb['next_obs_embedding']
                    # all_features = crafter_utils.obs_to_features(all_next_obs)
                    # metrics['feature_divergences'] = crafter_vis.feature_divergences(all_features, all_skills)

                    # tsne_features = crafter_utils.tsne_embeddings(all_next_obs_embeddings)
                    # metrics['embeddings/by_skill'] = crafter_vis.plot_embeddings_by_skill(tsne_features, all_skills, config.skill_size)
                    # metrics['embeddings/by_timestep'] = crafter_vis.plot_embeddings_by_timestep(tsne_features, all_timesteps)

                    achievement_counts = onp.zeros_like(achievement_counts)
                    action_skill_mtx = onp.zeros_like(action_skill_mtx)

        if log:
            wandb.log(metrics, step=t_step*config.num_workers)
            metrics.clear()
        
        if log and config.save_checkpoints and (t_step % config.save_freq == 0 or t_step == config.steps - 1):
            jnp.savez(os.path.join(os.getcwd(), 'data', run_name),
                t_step=t_step,
                qlocal_params=qlocal_params,
                qtarget_params=qtarget_params,
                discrim_params=discrim_params)

        obs, state = next_obs, next_state
        if t_step > config.warmup_steps:
            eps = max(eps * (config.eps_decay ** config.num_workers), config.eps_end)
        score_gt = score_gt.at[done_idx].set(0.0)
        score_pr = score_pr.at[done_idx].set(0.0)
        steps_per_ep = steps_per_ep.at[done_idx].set(0)
        key, skill_update_key = random.split(key)
        skill_idx = jnp.where(
            done,
            random.randint(skill_update_key, minval=0, maxval=config.skill_size, shape=(config.num_workers,)),
            skill_idx
        )


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
        if 'name' not in config:
            run_name = '{}_{}_{}_{}_{}'.format(config.env_name, config.exp_type, config.skill_size, config.embedding_type, int(datetime.now().timestamp()))
        else:
            run_name = '{}_{}'.format(config.name, int(datetime.now().timestamp()))
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
    # config.num_workers = len(devices)
    config.num_workers = 100

    key = random.PRNGKey(config.seed)
    train(key, config, run_name, log=not args.nolog)
    run.finish()
