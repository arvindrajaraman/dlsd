from flax import linen as nn
from PIL import Image, ImageDraw
import gym
import icecream as ic
import imageio
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as onp
import time
import wandb

from lunar_utils import normalize_freq_matrix

colormaps = ['Reds', 'Greens', 'Blues', 'Purples', 'Oranges']

def plot_visits(bins, x_min, x_max, y_min, y_max, skill_idx=None, log=False):
    if log:
        bins = jnp.where(bins != 0, jnp.log(bins + 1e-9), 0)
        label = 'log(Visitation)'
    else:
        label = 'Visitation'
    plt.close()
    plt.imshow(bins, cmap=colormaps[skill_idx % len(colormaps)] if skill_idx is not None else 'Greys',
               interpolation='nearest', extent=[x_min, x_max, y_min, y_max], origin='lower')
    plt.colorbar(label=label)
    plt.title(f'{label} Distribution')
    return wandb.Image(plt)

def goal_skill_heatmap(freq_matrix):
    plt.close()
    row_labels = [f'skill {i}' for i in range(3)]
    col_labels = [f'goal {i}' for i in range(3)]

    plt.imshow(freq_matrix)
    plt.xticks(jnp.arange(len(col_labels)), labels=col_labels)
    plt.yticks(jnp.arange(len(row_labels)), labels=row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            plt.text(j, i, freq_matrix[i, j], ha="center", va="center", color="w", bbox=dict(facecolor='black', alpha=0.2))

    plt.title('Confusion matrix of (skills, goals)')
    plt.tight_layout()
    plt.colorbar()
    return wandb.Image(plt)

def action_skill_heatmap(freq_matrix):
    plt.close()
    row_labels = [f'skill {i}' for i in range(3)]
    col_labels = ['no op', 'fire left', 'fire main', 'fire right']

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

def plot_pred_landscape(discrim, discrim_params, embedding_fn):
    plt.close()
    xs = onp.arange(-1.0, 1.005, 0.005)
    ys = onp.arange(0.0, 1.005, 0.005)

    X, Y = onp.meshgrid(xs, ys)
    points = onp.stack((X, Y), axis=2).reshape(-1, 2)
    points = onp.concatenate([points, onp.zeros((points.shape[0], 6))], axis=1)
    embeddings, _ = embedding_fn(points)

    skill_logits = discrim.apply(discrim_params, embeddings).reshape((X.shape[0], X.shape[1], 3))
    skill_probs = nn.activation.softmax(skill_logits)
    plt.pcolormesh(X, Y, skill_probs, shading='gouraud')

    # Add legend with red, green, blue
    plt.legend(handles=[mpatches.Patch(color='#ff0000', label='Skill 0'),
                       mpatches.Patch(color='#00ff00', label='Skill 1'),
                       mpatches.Patch(color='#0000ff', label='Skill 2')])
    
    plt.title('Prediction Landscape')
    plt.xlabel('X')
    plt.ylabel('Y')

    return wandb.Image(plt)

action_names = ['no op', 'fire left', 'fire main', 'fire right']

def gather_rollouts_by_skill(qlocal, qlocal_params, discrim, discrim_params, embedding_fn,
                             skill_size, num_rollouts, max_steps_per_rollout):
    test_env = gym.make('LunarLander-v2')
    skill_rollouts = []
    for skill_idx in range(skill_size):
        skill = jax.nn.one_hot(skill_idx, skill_size).reshape(1, -1)
        rollouts = []
        for rollout_num in range(num_rollouts):
            print(f'Visualizing rollout {rollout_num} of skill {skill_idx}')
            obs = test_env.reset()

            rollout_filepath = f'video_buffer/{time.time()}.mp4'
            writer = imageio.get_writer(rollout_filepath, fps=16)
            for t in range(max_steps_per_rollout):
                qvalues = qlocal.apply(qlocal_params, obs.reshape(1, -1), skill)
                action_probs = jax.nn.softmax(qvalues)[0]
                action = jnp.argmax(qvalues, axis=-1).item()

                next_obs, reward_gt, done, info = test_env.step(action)
                next_obs_embedding, next_obs_sentence = embedding_fn(next_obs.reshape(1, -1))
                discrim_logits = discrim.apply(discrim_params, next_obs_embedding)
                discrim_probs = jax.nn.softmax(discrim_logits)[0]

                next_obs_x, next_obs_y, _, _, next_obs_angle, _, _, _ = next_obs
                next_obs_angle = onp.degrees(next_obs_angle)

                fig, ax = plt.subplots()
                canvas = fig.canvas
                ax.add_patch(mpatches.Rectangle((next_obs_x, next_obs_y), 0.11, 0.05, color='#6E50D2', angle=next_obs_angle))
                ax.set_xlim([-1, 1])
                ax.set_ylim([0, 2])
                fig.set_facecolor('black')
                ax.axis('off')
                fig.tight_layout()
                canvas.draw()
                plt.close()

                img_flat = onp.frombuffer(canvas.tostring_rgb(), dtype='uint8')
                img = img_flat.reshape(*reversed(canvas.get_width_height()), 3)
                img = Image.fromarray(img).convert('RGB')

                action_name = action_names[action]
                action_prob = round(action_probs[action].item(), 4)
                discrim_prob = round(discrim_probs[skill_idx].item(), 4)
                reward_gt = round(reward_gt, 4)
                color = (255, 255, 255)

                draw = ImageDraw.Draw(img)
                draw.text((10, 410), f"t={t}", color, font_size=24, stroke_width=1)
                draw.text((10, 10), f"Action: {action_name}", color, font_size=24, stroke_width=1)
                draw.text((10, 40), f"Action Prob: {action_prob}", color, font_size=24,
                        stroke_width=1)
                draw.text((300, 10), f"Skill {skill_idx} Prob: {discrim_prob}", color, font_size=24,
                        stroke_width=1)
                draw.text((300, 40), f"Reward GT: {reward_gt}", color, font_size=24, stroke_width=1)

                writer.append_data(onp.array(img))

                obs = next_obs
                if done:
                    break
            writer.close()
            rollouts.append(rollout_filepath)
        skill_rollouts.append(rollouts)
    return skill_rollouts

def gather_rollouts(qlocal, qlocal_params, num_rollouts, max_steps_per_rollout):
    test_env = gym.make('LunarLander-v2')
    rollouts = []
    for rollout_num in range(num_rollouts):
        print(f'Visualizing rollout {rollout_num}')
        obs = test_env.reset()

        rollout_filepath = f'video_buffer/{time.time()}.mp4'
        writer = imageio.get_writer(rollout_filepath, fps=16)
        for t in range(max_steps_per_rollout):            
            qvalues = qlocal.apply(qlocal_params, obs.reshape(1, -1))
            action_probs = jax.nn.softmax(qvalues)[0]
            action = jnp.argmax(qvalues, axis=-1).item()

            next_obs, reward_gt, done, info = test_env.step(action)
            next_obs_x, next_obs_y, _, _, next_obs_angle, _, _, _ = next_obs
            next_obs_angle = onp.degrees(next_obs_angle)

            fig, ax = plt.subplots()
            canvas = fig.canvas
            ax.add_patch(mpatches.Rectangle((next_obs_x, next_obs_y), 0.11, 0.05, color='#6E50D2', angle=next_obs_angle))
            ax.set_xlim([-1, 1])
            ax.set_ylim([0, 2])
            fig.set_facecolor('black')
            ax.axis('off')
            fig.tight_layout()
            canvas.draw()
            plt.close()

            img_flat = onp.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            img = img_flat.reshape(*reversed(canvas.get_width_height()), 3)
            img = Image.fromarray(img).convert('RGB')

            action_name = action_names[action]
            action_prob = round(action_probs[action].item(), 4)
            reward_gt = round(reward_gt, 4)
            color = (255, 255, 255)

            draw = ImageDraw.Draw(img)
            draw.text((10, 410), f"t={t}", color, font_size=24, stroke_width=1)
            draw.text((10, 10), f"Action: {action_name}", color, font_size=24, stroke_width=1)
            draw.text((10, 40), f"Action Prob: {action_prob}", color, font_size=24,
                    stroke_width=1)
            draw.text((300, 10), f"Reward GT: {reward_gt}", color, font_size=24, stroke_width=1)

            writer.append_data(onp.array(img))

            obs = next_obs
            if done:
                break
        writer.close()
        rollouts.append(rollout_filepath)
    return rollouts
