import imageio
from PIL import Image, ImageDraw
import time

from craftax.craftax_env import make_craftax_env_from_name
from craftax.craftax_classic.renderer import render_craftax_pixels
import jax
from jax import random, vmap
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as onp
import wandb
from icecream import ic

from crafter_constants import achievement_labels, action_names, feature_names
import crafter_utils

def action_skill_heatmap(freq_matrix):
    plt.close()
    row_labels = [f'skill {i}' for i in range(3)]
    col_labels = action_names

    plt.figure(figsize=(4, 7))
    plt.imshow(freq_matrix)
    plt.xticks(jnp.arange(len(col_labels)), labels=col_labels)
    plt.yticks(jnp.arange(len(row_labels)), labels=row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            plt.text(j, i, freq_matrix[i, j], ha="center", va="center", color="w", bbox=dict(facecolor='black', alpha=0.2))

    plt.title('Confusion matrix of (skills, actions)')
    plt.tight_layout()
    plt.colorbar()
    return wandb.Image(plt)


def achievement_counts_heatmap(achievement_counts, skill_size):
    plt.close()

    skill_labels = [f'Skill {i}' for i in range(skill_size)]

    plt.figure(figsize=(6, 9))
    plt.imshow(achievement_counts.T)
    plt.xticks(range(skill_size), skill_labels, rotation=45)
    plt.yticks(range(len(achievement_labels)), achievement_labels)
    plt.title('Achievement Counts by Skill')
    plt.tight_layout()
    plt.colorbar()

    for i in range(len(achievement_labels)):
        for j in range(skill_size):
            plt.text(j, i, round(achievement_counts[j, i], 2), ha="center", va="center", color="w")
    
    return wandb.Image(plt)

def gather_rollouts_by_skill(key, qlocal, qlocal_params, discrim, discrim_params, embedding_fn,
                             skill_size, num_rollouts, max_steps_per_rollout):
    test_env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', auto_reset=False)
    env_params = test_env.default_params
    skill_rollouts = []
    for skill_idx in range(skill_size):
        skill = jax.nn.one_hot(skill_idx, skill_size).reshape(1, -1)
        rollouts = []
        for rollout_num in range(num_rollouts):
            print(f'Visualizing rollout {rollout_num} of skill {skill_idx}')
            key, env_init_key = random.split(key, num=2)
            obs, state = test_env.reset(env_init_key, env_params)

            rollout_filepath = f'video_buffer/{time.time()}.mp4'
            writer = imageio.get_writer(rollout_filepath, fps=8)
            for t in range(max_steps_per_rollout):
                key, env_step_key = random.split(key, num=2)
                
                qvalues = qlocal.apply(qlocal_params, obs.reshape(1, -1), skill)
                action_probs = jax.nn.softmax(qvalues)[0]
                action = jnp.argmax(qvalues, axis=-1).item()

                next_obs, next_state, reward_gt, done, info = test_env.step(env_step_key, state, action,
                                                                    env_params)
                next_obs_embedding, next_obs_sentence = embedding_fn(next_obs.reshape(1, -1))
                discrim_logits = discrim.apply(discrim_params, next_obs_embedding)
                discrim_probs = jax.nn.softmax(discrim_logits)[0]

                frame = render_craftax_pixels(state, 64) / 255.0
                img = Image.fromarray(onp.uint8(frame * 255)).convert('RGB')

                action_name = action_names[action]
                action_prob = round(action_probs[action].item(), 4)
                discrim_prob = round(discrim_probs[skill_idx].item(), 4)
                reward_gt = round(reward_gt.item(), 4)
                color = (255, 255, 255)

                tuples = list(zip(action_names, qvalues[0].tolist()))
                tuples.sort(key=lambda x: x[1])
                sorted_action_names, sorted_action_values = [x[0] for x in tuples], [x[1] for x in tuples]
                fig, ax = plt.subplots()
                canvas = fig.canvas
                fig.set_size_inches(5, 4)
                bars = ax.barh(sorted_action_names, sorted_action_values, color='lightgrey')
                ax.bar_label(bars, fmt='{:.2f}', label_type='edge', color='black')
                ax.set_title('Q-Values')
                ax.set_xlim(min(sorted_action_values) * 0.95, max(sorted_action_values) * 1.05)
                fig.tight_layout()
                canvas.draw()
                plt.close()
                plt_img = onp.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                plt_img = plt_img.reshape(*reversed(canvas.get_width_height()), 3)
                plt_img = Image.fromarray(plt_img).convert('RGB')

                full_img = Image.new('RGBA', (576, 976), (255, 255, 255, 255))
                full_img.paste(img, (0, 0))
                full_img.paste(plt_img, (0, 576))

                draw = ImageDraw.Draw(full_img)
                draw.text((10, 410), f"t={t}", color, font_size=24, stroke_width=1)
                draw.text((10, 10), f"Action: {action_name}", color, font_size=24, stroke_width=1)
                draw.text((10, 40), f"Action Prob: {action_prob}", color, font_size=24,
                        stroke_width=1)
                draw.text((300, 10), f"Skill {skill_idx} Prob: {discrim_prob}", color, font_size=24,
                        stroke_width=1)
                draw.text((300, 40), f"Reward GT: {reward_gt}", color, font_size=24, stroke_width=1)

                writer.append_data(onp.array(full_img))

                obs, state = next_obs, next_state
                if done:
                    break
            writer.close()
            rollouts.append(rollout_filepath)
        skill_rollouts.append(rollouts)
    return skill_rollouts

def gather_rollouts(key, qlocal, qlocal_params, num_rollouts, max_steps_per_rollout):
    test_env = make_craftax_env_from_name('Craftax-Classic-Symbolic-v1', auto_reset=False)
    env_params = test_env.default_params
    rollouts = []
    for rollout_num in range(num_rollouts):
        print(f'Visualizing rollout {rollout_num}')
        key, env_init_key = random.split(key, num=2)
        obs, state = test_env.reset(env_init_key, env_params)

        rollout_filepath = f'video_buffer/{time.time()}.mp4'
        writer = imageio.get_writer(rollout_filepath, fps=8)
        for t in range(max_steps_per_rollout):
            key, env_step_key = random.split(key, num=2)
            
            qvalues = qlocal.apply(qlocal_params, obs.reshape(1, -1))
            action_probs = jax.nn.softmax(qvalues)[0]
            action = jnp.argmax(qvalues, axis=-1).item()

            next_obs, next_state, reward_gt, done, info = test_env.step(env_step_key, state, action,
                                                                env_params)

            frame = render_craftax_pixels(state, 64) / 255.0
            img = Image.fromarray(onp.uint8(frame * 255)).convert('RGB')

            action_name = action_names[action]
            action_prob = round(action_probs[action].item(), 4)
            reward_gt = round(reward_gt.item(), 4)
            color = (255, 255, 255)

            tuples = list(zip(action_names, qvalues[0].tolist()))
            tuples.sort(key=lambda x: x[1])
            sorted_action_names, sorted_action_values = [x[0] for x in tuples], [x[1] for x in tuples]
            fig, ax = plt.subplots()
            canvas = fig.canvas
            fig.set_size_inches(5, 4)
            bars = ax.barh(sorted_action_names, sorted_action_values, color='lightgrey')
            ax.bar_label(bars, fmt='{:.2f}', label_type='edge', color='black')
            ax.set_title('Q-Values')
            ax.set_xlim(min(sorted_action_values) * 0.95, max(sorted_action_values) * 1.05)
            fig.tight_layout()
            canvas.draw()
            plt.close()
            plt_img = onp.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            plt_img = plt_img.reshape(*reversed(canvas.get_width_height()), 3)
            plt_img = Image.fromarray(plt_img).convert('RGB')

            full_img = Image.new('RGBA', (576, 976), (255, 255, 255, 255))
            full_img.paste(img, (0, 0))
            full_img.paste(plt_img, (0, 576))

            draw = ImageDraw.Draw(full_img)
            draw.text((10, 410), f"t={t}", color, font_size=24, stroke_width=1)
            draw.text((10, 10), f"Action: {action_name}", color, font_size=24, stroke_width=1)
            draw.text((10, 40), f"Action Prob: {action_prob}", color, font_size=24,
                    stroke_width=1)
            draw.text((300, 10), f"Reward GT: {reward_gt}", color, font_size=24, stroke_width=1)

            writer.append_data(onp.array(full_img))

            obs, state = next_obs, next_state
            if done:
                break
        writer.close()
        rollouts.append(rollout_filepath)
    return rollouts

def feature_divergences(features, skills):
    def kl_divergence(mu1, sigma1, mu2, sigma2):
        return jnp.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5
    
    def log_average_divergence(means, stds):
        n = len(means)
        kl_matrix = jnp.zeros((n, n))

        def pairwise_kl(i, j):
            return kl_divergence(means[i], stds[i], means[j], stds[j])

        kl_matrix = vmap(pairwise_kl, (0, None))(jnp.arange(n), jnp.arange(n))
        avg_div = jnp.mean(kl_matrix)
        return jnp.log(avg_div + 1e-9)
    
    skill_idxs = jnp.argmax(skills, axis=1)
    grouped_features, _ = crafter_utils.separate_features(features, skill_idxs)

    all_means = []
    all_stds = []
    for i in range(len(grouped_features)):
        feature_means = jnp.mean(grouped_features[i], axis=0)
        feature_stds = jnp.std(grouped_features[i], axis=0)

        all_means.append(feature_means)
        all_stds.append(feature_stds)
    all_means = jnp.array(all_means)
    all_stds = jnp.array(all_stds)

    # For any features where there is no variance (i.e. all the feature values are the same), make the variance 1 so divergences can be calculated.
    all_stds = jnp.where(jnp.isclose(all_stds, 0), 1, all_stds)

    avg_divs = vmap(lambda i: log_average_divergence(all_means[:, i], all_stds[:, i]))(jnp.arange(grouped_features[0].shape[1]))
    colors = ['blue'] * 17 + ['red'] * 4 + ['green'] * 12 + ['orange'] * 10

    pairs = zip(feature_names, avg_divs, colors)
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    sorted_feature_names, avg_divs, colors = zip(*pairs)

    plt.figure(figsize=(5, 8))
    plt.barh(y=sorted_feature_names, width=avg_divs, color=colors)
    plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
    plt.title('Feature Divergence Across Skills')
    plt.xlabel('Log(Average KL-Divergence)')
    plt.ylabel('Feature')
    plt.legend(handles=[mpatches.Patch(color='blue', label='Block'),
                        mpatches.Patch(color='red', label='Mob'),
                        mpatches.Patch(color='green', label='Inventory'),
                        mpatches.Patch(color='orange', label='Other Metadata')])
    plt.tight_layout()

    return wandb.Image(plt)


def plot_embeddings_by_skill(features_embedded, skills, skill_size):
    print('Plotting embeddings by skill...')
    skill_idxs = jnp.argmax(skills, axis=1)
    grouped_features, grouped_skill_idxs = crafter_utils.separate_features(features_embedded, skill_idxs)
    grouped_features = jnp.concatenate(grouped_features, axis=0)
    grouped_skill_idxs = jnp.concatenate(grouped_skill_idxs, axis=0)
    
    cmap = mpl.colormaps['rainbow']
    colors = jnp.array(cmap(onp.linspace(0, 1, skill_size)))
    colors = colors.at[:, -1].set(0.2)
    color_assignments = colors[grouped_skill_idxs]

    plt.figure(figsize=(8, 8))
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=color_assignments)
    plt.title('t-SNE of State Embeddings by Skill')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.tight_layout()

    handles = [mpatches.Patch(color=onp.array(colors[i]), label=f'Skill {i}') for i in range(skill_size)]
    plt.legend(handles=handles, loc='lower right')
    return wandb.Image(plt)

def plot_embeddings_by_timestep(features_embedded, timesteps):
    print('Plotting embeddings by timestep...')
    timestep_bins = crafter_utils.bin_timesteps(timesteps)
    grouped_features, grouped_bins = crafter_utils.separate_features(features_embedded, timestep_bins)
    grouped_features = jnp.concatenate(grouped_features, axis=0)
    grouped_bins = jnp.concatenate(grouped_bins, axis=0)
    
    cmap = mpl.colormaps['plasma']
    colors = jnp.array(cmap(onp.linspace(0, 1, 8)))
    colors = colors.at[:, -1].set(0.2)
    color_assignments = colors[grouped_bins]

    plt.figure(figsize=(8, 8))
    plt.scatter(features_embedded[:, 0], features_embedded[:, 1], c=color_assignments)
    plt.title('t-SNE of State Embeddings by Timestep')
    plt.xlabel('t-SNE Feature 1')
    plt.ylabel('t-SNE Feature 2')
    plt.tight_layout()

    handles = [mpatches.Patch(color=onp.array(colors[0]), label=f't=0-50'),
               mpatches.Patch(color=onp.array(colors[1]), label=f't=50-100'),
               mpatches.Patch(color=onp.array(colors[2]), label=f't=100-150'),
               mpatches.Patch(color=onp.array(colors[3]), label=f't=150-200'),
               mpatches.Patch(color=onp.array(colors[4]), label=f't=200-250'),
               mpatches.Patch(color=onp.array(colors[5]), label=f't=250-300'),
               mpatches.Patch(color=onp.array(colors[6]), label=f't=300-400'),
               mpatches.Patch(color=onp.array(colors[7]), label=f't=400+')]
    plt.legend(handles=handles, loc='lower right')
    return wandb.Image(plt)
