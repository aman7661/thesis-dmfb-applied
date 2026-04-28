"""
train.py — Single-agent PPO training for DMFB droplet routing.

Paper Section 5.4:
  - Train on healthy mode until performance matches baseline
  - Then evaluate on degrading mode
  - Use 8 concurrent environments (optimal PPO setting from paper)
  - Each epoch = 20,000 timesteps
  - Full convergence at ~800K timesteps
"""

import gym
import dmfb_env
import argparse
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    EvalCallback, CheckpointCallback, BaseCallback
)
from dmfb_env.models.cnn_policy import DMFBCNNPolicy

def make_env(size, mode, seed, utilization_lambda=0.0, persist_usage_across_episodes=False):
    def _init():
        env = gym.make(
            'DMFB-v0',
            size=size,
            mode=mode,
            utilization_lambda=utilization_lambda,
            persist_usage_across_episodes=persist_usage_across_episodes,
        )
        env.seed(seed)
        return env
    return _init

class TrainingProgressCallback(BaseCallback):
    """
    Logs training progress per epoch.
    Paper uses epochs of 20,000 timesteps.
    """
    def __init__(self, epoch_size=20000, verbose=1):
        super().__init__(verbose)
        self.epoch_size = epoch_size
        self.epoch_rewards = []
        self.current_epoch_rewards = []

    def _on_step(self):
        if self.locals.get('rewards') is not None:
            self.current_epoch_rewards.extend(self.locals['rewards'])

        if self.num_timesteps % self.epoch_size == 0:
            if self.current_epoch_rewards:
                mean_reward = np.mean(self.current_epoch_rewards)
                self.epoch_rewards.append(mean_reward)
                if self.verbose:
                    epoch_num = self.num_timesteps // self.epoch_size
                    print(f"Epoch {epoch_num:3d} | "
                          f"Mean Reward: {mean_reward:7.3f} | "
                          f"Timesteps: {self.num_timesteps:,}")
                self.current_epoch_rewards = []
        return True

def train(
    size=10,
    total_timesteps=1_000_000,
    n_envs=8,
    save_dir='log/',
    utilization_lambda=0.0,
    persist_usage_across_episodes=False,
):
    """
    Main training function.

    Args:
      size            : DMFB grid size (N for NxN array)
      total_timesteps : Total training timesteps
      n_envs          : Parallel environments
                        Paper found 8 optimal — Section 5.4
      save_dir        : Directory to save models and logs
      utilization_lambda : Fairness shaping weight (0 = paper reward only)
      persist_usage_across_episodes : If True, usage matrix persists across resets (bioassay)
    """
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training PPO on {size}x{size} DMFB")
    print(f"Environments: {n_envs} | Timesteps: {total_timesteps:,}")
    print(f"utilization_lambda: {utilization_lambda}")
    print(f"{'='*60}\n")

    # Vectorized training environments — 8 concurrent (paper's optimal)
    train_env = SubprocVecEnv([
        make_env(
            size=size,
            mode='healthy',
            seed=i,
            utilization_lambda=utilization_lambda,
            persist_usage_across_episodes=persist_usage_across_episodes,
        )
        for i in range(n_envs)
    ])
    train_env = VecMonitor(train_env)

    eval_env = gym.make(
        'DMFB-v0',
        size=size,
        mode='healthy',
        utilization_lambda=utilization_lambda,
        persist_usage_across_episodes=persist_usage_across_episodes,
    )

    model = PPO(
        policy='CnnPolicy',
        env=train_env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        tensorboard_log=save_dir,
        policy_kwargs={
            "features_extractor_class": DMFBCNNPolicy,
            "features_extractor_kwargs": {"features_dim": 8},
            "net_arch": []
        }
    )

    progress_cb = TrainingProgressCallback(epoch_size=20000)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=save_dir,
        eval_freq=20000,
        n_eval_episodes=50,
        deterministic=True,
        render=False,
        verbose=1
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=100000,
        save_path=save_dir,
        name_prefix='checkpoint'
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[progress_cb, eval_cb, checkpoint_cb],
        reset_num_timesteps=True
    )

    model.save(os.path.join(save_dir, f'final_model_{size}x{size}'))
    print(f"\nTraining complete. Model saved to {save_dir}")

    train_env.close()
    eval_env.close()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--timesteps', type=int, default=1_000_000)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='log/')
    parser.add_argument('--utilization_lambda', type=float, default=0.0)
    parser.add_argument(
        '--persist_usage_across_episodes',
        action='store_true',
        help='Keep usage matrix across episode resets (bioassay-style)',
    )
    args = parser.parse_args()

    train(
        size=args.size,
        total_timesteps=args.timesteps,
        n_envs=args.n_envs,
        save_dir=args.save_dir,
        utilization_lambda=args.utilization_lambda,
        persist_usage_across_episodes=args.persist_usage_across_episodes,
    )
