#!/usr/bin/python
"""
Training and paper-style evaluation for DMFBEnv (PPO + CNN).

Related work — degradation-aware droplet routing is formulated as RL in:
  T.-C. Liang et al., "Dynamic Adaptation Using Deep Reinforcement Learning
  for Digital Microfluidic Biochips," ACM TODAES 29(2), Article 24, 2024.
  https://doi.org/10.1145/3633458

That work models electrode health d in [0,1] with transition success ~ d
(see their Sec. 2.2), motivating simulator-style training before deployment.
This codebase adds an explicit *usage-penalty* (lambda) to spread actuations
across pins—aligned with synthesis goals of avoiding overuse of a few
electrodes they discuss (their Sec. 1.2, citing prior synthesis methods).

Changing the reward (lambda) changes the MDP: compare policies trained
with matched seeds and timestep budgets (lambda=0 vs lambda>0).
"""

import csv
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import OldRouter
from my_net import MyCnnExtractor
from envs.dmfb import DMFBEnv

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dmfb_unwrapped(vec_env, index=0):
    """Reach DMFBEnv through Gym / SB3 Monitor wrappers."""
    e = vec_env.envs[index]
    while hasattr(e, "unwrapped"):
        inner = e.unwrapped
        if isinstance(inner, DMFBEnv):
            return inner
        if inner is e:
            break
        e = inner
    raise TypeError("DMFBEnv not found under vec_env.envs[%s]" % index)


def legacyReward(env, b_path=False):
    router = OldRouter(env)
    return router.getReward(b_path)


def evaluate_policy_detailed(
        model, vec_env, n_eval_episodes=50, b_path=False,
        deterministic=True):
    """
    Eval mean return, legacy (once per episode at start), usage std per
    episode, and mean usage heatmap averaged over eval episodes.
    """
    episode_rewards = []
    episode_usage_stds = []
    legacies = []
    heat_sum = None

    for _ in range(n_eval_episodes):
        obs = vec_env.reset()
        uw = get_dmfb_unwrapped(vec_env, 0)
        legacy_r = legacyReward(uw, b_path)
        done = False
        episode_reward = 0.0
        while not done:
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, rewards, dones, infos = vec_env.step(action)
            episode_reward += float(rewards[0])
            done = bool(dones[0])

        uw = get_dmfb_unwrapped(vec_env, 0)
        ustd = uw._last_episode_usage_std
        umap = uw._last_episode_usage_map
        episode_usage_stds.append(float(ustd) if ustd is not None else 0.0)
        if umap is not None:
            if heat_sum is None:
                heat_sum = np.zeros_like(umap, dtype=np.float64)
            heat_sum += umap.astype(np.float64)
        episode_rewards.append(episode_reward)
        legacies.append(legacy_r)

    mean_h = (heat_sum / n_eval_episodes) if heat_sum is not None else None
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards, ddof=1))
        if len(episode_rewards) > 1 else 0.0,
        "mean_usage_std": float(np.mean(episode_usage_stds))
        if episode_usage_stds else 0.0,
        "std_usage_std": float(np.std(episode_usage_stds, ddof=1))
        if len(episode_usage_stds) > 1 else 0.0,
        "mean_legacy": float(np.mean(legacies)),
        "mean_usage_heatmap": mean_h,
    }


def _append_csv_row(path, fieldnames, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)


def _publication_rc():
    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "figure.figsize": (6.5, 4.0),
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def save_publication_figure(fig, basepath_no_ext):
    """PNG (raster) + PDF/SVG (vector) for camera-ready figures."""
    os.makedirs(os.path.dirname(basepath_no_ext) or ".", exist_ok=True)
    fig.savefig(basepath_no_ext + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".pdf", bbox_inches="tight")
    fig.savefig(basepath_no_ext + ".svg", bbox_inches="tight")


def runAnExperiment(
        env,
        model=None,
        num_iterations=50,
        num_steps=2000,
        policy_steps=128,
        b_path=False,
        seed=0,
        electrode_lambda=0.0,
        csv_path=None,
        tensorboard_log=None,
        n_eval_episodes=50):
    """
    Train/eval loop. Logs one CSV row per (iteration, eval) if csv_path set.
    Returns sliced learning curves and last evaluate_policy_detailed dict.
    """
    fieldnames = [
        "seed", "electrode_lambda", "iteration", "timesteps_cumulative",
        "eval_mean_return", "eval_std_return", "eval_mean_usage_std",
        "eval_std_usage_std", "legacy_mean", "train_wall_s",
    ]

    if model is None:
        print("--- Initializing PPO Model with Custom Extractor ---")
        policy_kwargs = dict(
            features_extractor_class=MyCnnExtractor,
            features_extractor_kwargs=dict(features_dim=256),
        )
        tb_kw = {}
        if tensorboard_log:
            os.makedirs(tensorboard_log, exist_ok=True)
            tb_kw["tensorboard_log"] = tensorboard_log
        model = PPO(
            "CnnPolicy",
            env,
            n_steps=policy_steps,
            policy_kwargs=policy_kwargs,
            verbose=0,
            **tb_kw,
        )
        print("--- Model Initialization Complete ---\n")

    agent_rewards = []
    old_rewards = []
    episodes = []
    last_eval = None

    for i in range(num_iterations + 1):
        print("  [Iteration %s/%s] Training for %s timesteps..."
              % (i, num_iterations, num_steps))
        t0 = time.time()
        model.learn(total_timesteps=num_steps)
        train_wall = time.time() - t0
        cum_steps = (i + 1) * num_steps

        print("  [Iteration %s/%s] Evaluating (%s episodes)..."
              % (i, num_iterations, n_eval_episodes))
        last_eval = evaluate_policy_detailed(
            model, model.get_env(), n_eval_episodes=n_eval_episodes,
            b_path=b_path)

        print("  --> Mean Reward: %.2f | Legacy: %.2f | Usage std (mean): %.4f\n"
              % (last_eval["mean_reward"], last_eval["mean_legacy"],
                 last_eval["mean_usage_std"]))

        if csv_path:
            _append_csv_row(csv_path, fieldnames, {
                "seed": seed,
                "electrode_lambda": electrode_lambda,
                "iteration": i,
                "timesteps_cumulative": cum_steps,
                "eval_mean_return": last_eval["mean_reward"],
                "eval_std_return": last_eval["std_reward"],
                "eval_mean_usage_std": last_eval["mean_usage_std"],
                "eval_std_usage_std": last_eval["std_usage_std"],
                "legacy_mean": last_eval["mean_legacy"],
                "train_wall_s": round(train_wall, 3),
            })

        agent_rewards.append(last_eval["mean_reward"])
        old_rewards.append(last_eval["mean_legacy"])
        episodes.append(i)

    agent_rewards = agent_rewards[-num_iterations:]
    old_rewards = old_rewards[-num_iterations:]
    episodes = episodes[:num_iterations]
    return agent_rewards, old_rewards, episodes, last_eval


def showIsGPU():
    if torch.cuda.is_available():
        print("### Training on GPUs... ###")
    else:
        print("### Training on CPUs... ###")


def plotAgentPerformance(
        a_rewards, o_rewards, size, env_info, b_path=False,
        out_dir="log", suffix=""):
    """Legacy min/max band across runs (supplementary)."""
    print("Saving performance plot (min/max band) for size %s..." % size)
    a_rewards = np.asarray(a_rewards, dtype=np.float64)
    o_rewards = np.asarray(o_rewards, dtype=np.float64)
    a_line = np.mean(a_rewards, axis=0)
    o_line = np.mean(o_rewards, axis=0)
    a_max = np.max(a_rewards, axis=0)
    a_min = np.min(a_rewards, axis=0)
    o_max = np.max(o_rewards, axis=0)
    o_min = np.min(o_rewards, axis=0)
    x = np.arange(len(a_line))

    _publication_rc()
    fig, ax = plt.subplots()
    ax.fill_between(x, a_min, a_max, color="tab:red", alpha=0.25,
                    label="Agent min/max")
    ax.fill_between(x, o_min, o_max, color="tab:blue", alpha=0.25,
                    label="Baseline min/max")
    ax.plot(x, a_line, color="tab:red", label="Agent mean")
    ax.plot(x, o_line, color="tab:blue", label="Baseline mean")
    ax.set_title("DMFB %s" % size)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Number of cycles" if b_path else "Score")
    ax.legend(loc="best")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, size + env_info + suffix)
    save_publication_figure(fig, base)
    plt.close(fig)
    print("Saved: %s.{png,pdf,svg}\n" % base)


def plot_learning_sem(
        curves_agent, curves_baseline, size, env_info, num_steps_per_iter,
        out_dir="log", suffix="_sem"):
    """
    Mean +- SEM across seeds; second x-axis = cumulative env steps
    (sample efficiency, as in RL reporting alongside Liang et al. training
    in simulation before hardware).
    """
    A = np.asarray(curves_agent, dtype=np.float64)
    B = np.asarray(curves_baseline, dtype=np.float64)
    n = A.shape[0]
    x = np.arange(A.shape[1])
    mean_a, mean_b = np.mean(A, axis=0), np.mean(B, axis=0)
    sem_a = np.std(A, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean_a)
    sem_b = np.std(B, axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean_b)
    steps = (x + 1) * float(num_steps_per_iter)

    _publication_rc()
    fig, ax = plt.subplots()
    ax.plot(x, mean_a, color="tab:red", label="RL agent (mean)")
    ax.fill_between(x, mean_a - sem_a, mean_a + sem_a,
                    color="tab:red", alpha=0.2)
    ax.plot(x, mean_b, color="tab:blue", label="Legacy baseline (mean)")
    ax.fill_between(x, mean_b - sem_b, mean_b + sem_b,
                    color="tab:blue", alpha=0.2)
    ax.set_xlabel("Training iteration")
    ax.set_ylabel("Eval mean return / legacy score")
    ax.set_title("DMFB %s (mean +/- SEM, n=%s seeds)" % (size, n))
    ax2 = ax.twiny()
    ax2.set_xlim(steps[0], steps[-1])
    ax2.set_xlabel("Cumulative training steps (approx.)")
    ax.legend(loc="best")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, size + env_info + suffix)
    save_publication_figure(fig, base)
    plt.close(fig)
    print("Saved SEM learning curve: %s.{png,pdf,svg}" % base)


def plot_std_vs_lambda(
        lambdas, mean_usage_std, sem_usage_std,
        mean_returns, sem_returns, out_path_base):
    """Primary democratization curve: usage spread vs lambda + return twin."""
    _publication_rc()
    fig, ax1 = plt.subplots()
    ax1.errorbar(lambdas, mean_usage_std, yerr=sem_usage_std,
                 fmt="o-", capsize=4, color="tab:purple",
                 label="Mean pin-usage std (eval)")
    ax1.set_xlabel(r"Electrode usage penalty $\lambda$")
    ax1.set_ylabel("Mean episode usage std (lower = more uniform)")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.errorbar(lambdas, mean_returns, yerr=sem_returns,
                 fmt="s--", capsize=4, color="tab:green",
                 label="Mean eval return")
    ax2.set_ylabel("Mean eval return")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    ax1.set_title(r"Usage fairness vs $\lambda$ (with return; cf. ACM TODAES '24 routing)")
    save_publication_figure(fig, out_path_base)
    plt.close(fig)


def plot_heatmap_pair(
        heat0, heat_lam, lam_star, w, l, out_path_base, vmax=None):
    """Side-by-side mean usage heatmaps, shared color scale."""
    if heat0 is None or heat_lam is None:
        return
    _publication_rc()
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    if vmax is None:
        vmax = max(float(np.max(heat0)), float(np.max(heat_lam)), 1e-6)
    im0 = axes[0].imshow(heat0, origin="upper", aspect="equal",
                         vmin=0.0, vmax=vmax)
    axes[0].set_title(r"$\lambda=0$")
    axes[1].imshow(heat_lam, origin="upper", aspect="equal",
                   vmin=0.0, vmax=vmax)
    axes[1].set_title(r"$\lambda=%s$" % lam_star)
    diff = heat_lam.astype(np.float64) - heat0.astype(np.float64)
    lim = max(abs(float(np.max(diff))), abs(float(np.min(diff))), 1e-9)
    axes[2].imshow(diff, origin="upper", aspect="equal",
                   cmap="coolwarm", vmin=-lim, vmax=lim)
    axes[2].set_title("Difference (lambda - baseline)")
    for ax in axes.flat:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im0, ax=[axes[0], axes[1]], shrink=0.85,
                 label="Mean actuations / cell")
    plt.tight_layout()
    save_publication_figure(fig, out_path_base)
    plt.close(fig)


def plot_final_bars(lambdas, mean_ret, std_ret, mean_u, std_u, out_path_base):
    _publication_rc()
    x = np.arange(len(lambdas))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, mean_ret, width, yerr=std_ret, capsize=3,
           label="Eval return", color="tab:green")
    ax.bar(x + width / 2, mean_u, width, yerr=std_u, capsize=3,
           label="Usage std", color="tab:purple")
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in lambdas])
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Value (mean +/- std across seeds)")
    ax.legend()
    ax.set_title("Endpoint comparison after training")
    save_publication_figure(fig, out_path_base)
    plt.close(fig)


def plot_pareto(mean_returns, mean_usage_stds, labels, out_path_base):
    _publication_rc()
    fig, ax = plt.subplots()
    for i, lab in enumerate(labels):
        ax.scatter(mean_usage_stds[i], mean_returns[i], s=80, label=lab)
    ax.set_xlabel("Mean usage std (lower is fairer)")
    ax.set_ylabel("Mean eval return")
    ax.set_title("Return vs pin-usage spread (Pareto-style)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    save_publication_figure(fig, out_path_base)
    plt.close(fig)


def write_summary_table(path, rows):
    """rows: list of dicts with keys lambda, mean_return, std_return_seeds, ..."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("Wrote summary table: %s" % path)


def expSeveralRuns(
        args, n_e, n_s, n_repeat=3, seeds=None,
        num_iterations=50, num_steps=2000,
        log_root="log", paper_subdir="paper"):
    """Multiple seeds, SEM plot, CSV per seed, optional TB."""
    if seeds is None:
        seeds = list(range(n_repeat))
    size = "%sx%s" % (args["w"], args["l"])
    env_info = "_m%s" % args["n_modules"]
    lam = float(args.get("electrode_usage_penalty_lambda", 0.0))
    out_paper = os.path.join(log_root, paper_subdir)
    os.makedirs(out_paper, exist_ok=True)

    a_rewards = []
    o_rewards = []
    for si, seed in enumerate(seeds):
        print("=" * 50)
        print(" Seed %s / %s  (lambda=%s)" % (si + 1, len(seeds), lam))
        print("=" * 50)
        set_all_seeds(seed)
        env = make_vec_env(DMFBEnv, n_envs=n_e, env_kwargs=args)
        showIsGPU()
        csv_path = os.path.join(
            out_paper, "train_log_w%s_m%s_lam%s_seed%s.csv"
            % (args["w"], args["n_modules"], lam, seed))
        tb = os.path.join(out_paper, "tb", "w%s_lam%s_seed%s"
                          % (args["w"], lam, seed))
        a_r, o_r, episodes, _last = runAnExperiment(
            env,
            num_iterations=num_iterations,
            num_steps=num_steps,
            policy_steps=n_s,
            seed=seed,
            electrode_lambda=lam,
            csv_path=csv_path,
            tensorboard_log=tb,
        )
        a_rewards.append(a_r)
        o_rewards.append(o_r)

    plotAgentPerformance(
        a_rewards, o_rewards, size, env_info, out_dir=log_root)
    plot_learning_sem(
        a_rewards, o_rewards, size, env_info, num_steps,
        out_dir=out_paper, suffix="_learning_sem")
    return np.asarray(a_rewards), np.asarray(o_rewards), episodes


def exp_lambda_sweep(
        base_args,
        lambdas,
        seeds,
        n_e=1,
        n_s=64,
        num_iterations=50,
        num_steps=2000,
        log_root="log",
        paper_subdir="paper"):
    """
    Train one policy per lambda (matched seeds & budget). Produces:
      - std vs lambda (+ return) figure
      - heatmap lambda=0 vs max positive lambda in sweep
      - endpoint bar chart, Pareto scatter, summary CSV
    """
    out_paper = os.path.join(log_root, paper_subdir)
    os.makedirs(out_paper, exist_ok=True)

    # Store final metrics and heatmaps per (lambda, seed)
    finals_ret = {lam: [] for lam in lambdas}
    finals_u = {lam: [] for lam in lambdas}
    heat_last = {lam: [] for lam in lambdas}

    curves_by_lam = {}

    for lam in lambdas:
        args = dict(base_args)
        args["electrode_usage_penalty_lambda"] = float(lam)
        seed_curves_a = []
        seed_curves_o = []
        for seed in seeds:
            set_all_seeds(seed)
            env = make_vec_env(DMFBEnv, n_envs=n_e, env_kwargs=args)
            showIsGPU()
            csv_path = os.path.join(
                out_paper, "train_log_w%s_m%s_lam%s_seed%s.csv"
                % (args["w"], args["n_modules"], lam, seed))
            tb = os.path.join(out_paper, "tb", "w%s_lam%s_seed%s"
                              % (args["w"], lam, seed))
            a_r, o_r, _ep, last_ev = runAnExperiment(
                env,
                num_iterations=num_iterations,
                num_steps=num_steps,
                policy_steps=n_s,
                seed=seed,
                electrode_lambda=lam,
                csv_path=csv_path,
                tensorboard_log=tb,
            )
            seed_curves_a.append(a_r)
            seed_curves_o.append(o_r)
            finals_ret[lam].append(last_ev["mean_reward"])
            finals_u[lam].append(last_ev["mean_usage_std"])
            if last_ev["mean_usage_heatmap"] is not None:
                heat_last[lam].append(last_ev["mean_usage_heatmap"])

        curves_by_lam[lam] = (np.asarray(seed_curves_a), np.asarray(seed_curves_o))

    # Aggregate across seeds
    mean_u = [float(np.mean(finals_u[lam])) for lam in lambdas]
    sem_u = [float(np.std(finals_u[lam], ddof=1) / np.sqrt(len(seeds)))
             if len(seeds) > 1 else 0.0 for lam in lambdas]
    mean_r = [float(np.mean(finals_ret[lam])) for lam in lambdas]
    sem_r = [float(np.std(finals_ret[lam], ddof=1) / np.sqrt(len(seeds)))
             if len(seeds) > 1 else 0.0 for lam in lambdas]

    plot_std_vs_lambda(
        lambdas, mean_u, sem_u, mean_r, sem_r,
        os.path.join(out_paper, "std_vs_lambda_twin"))

    # Heatmap: lambda=0 vs largest positive lambda when both exist
    pos = [l for l in lambdas if l > 0]
    lam_star = pos[-1] if pos else lambdas[0]
    h0_list = heat_last.get(0.0)
    hs_list = heat_last.get(lam_star)
    if h0_list and hs_list and lam_star != 0.0:
        h0 = np.mean(np.stack(h0_list, axis=0), axis=0)
        hs = np.mean(np.stack(hs_list, axis=0), axis=0)
        plot_heatmap_pair(
            h0, hs, lam_star, base_args["w"], base_args["l"],
            os.path.join(out_paper, "usage_heatmap_baseline_vs_lambda"))

    std_ret = [float(np.std(finals_ret[lam], ddof=1))
               if len(seeds) > 1 else 0.0 for lam in lambdas]
    std_u = [float(np.std(finals_u[lam], ddof=1))
             if len(seeds) > 1 else 0.0 for lam in lambdas]
    plot_final_bars(
        lambdas, mean_r, std_ret, mean_u, std_u,
        os.path.join(out_paper, "endpoint_bars_return_usagestd"))

    plot_pareto(
        mean_r, mean_u, [r"$\lambda=%s$" % l for l in lambdas],
        os.path.join(out_paper, "pareto_return_vs_usagestd"))

    summary_rows = []
    total_steps = num_iterations * num_steps
    for lam in lambdas:
        summary_rows.append({
            "electrode_lambda": lam,
            "mean_final_return": float(np.mean(finals_ret[lam])),
            "std_final_return_across_seeds": float(
                np.std(finals_ret[lam], ddof=1))
            if len(seeds) > 1 else 0.0,
            "mean_final_usage_std": float(np.mean(finals_u[lam])),
            "std_final_usage_std_across_seeds": float(
                np.std(finals_u[lam], ddof=1))
            if len(seeds) > 1 else 0.0,
            "total_train_steps_per_run": total_steps,
        })
    write_summary_table(
        os.path.join(out_paper, "summary_lambda_sweep.csv"), summary_rows)

    # Learning curves for lambda=0 vs first positive lambda (SEM)
    size = "%sx%s" % (base_args["w"], base_args["l"])
    env_info = "_m%s" % base_args["n_modules"]
    if 0.0 in curves_by_lam and pos:
        A0, B0 = curves_by_lam[0.0]
        A1, B1 = curves_by_lam[pos[0]]
        _publication_rc()
        fig, ax = plt.subplots()
        x = np.arange(A0.shape[1])
        for name, A, col in [
                (r"$\lambda=0$", A0, "tab:blue"),
                (r"$\lambda=%s$" % pos[0], A1, "tab:red")]:
            m = np.mean(A, axis=0)
            sem = np.std(A, axis=0, ddof=1) / np.sqrt(len(seeds))
            ax.plot(x, m, color=col, label=name + " agent")
            ax.fill_between(x, m - sem, m + sem, color=col, alpha=0.2)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Eval mean return")
        ax.set_title("RL agent learning: baseline vs usage penalty")
        ax.legend()
        ax2 = ax.twiny()
        ax2.set_xlim(num_steps, num_steps * len(x))
        ax2.set_xlabel("Cumulative training steps (approx.)")
        save_publication_figure(
            fig, os.path.join(out_paper, "learning_lambda0_vs_positive"))
        plt.close(fig)

    return curves_by_lam, summary_rows


if __name__ == "__main__":
    print("### Starting train.py ###")
    # Default grid size consistent with paper-style N x M arrays (tune freely).
    sizes = [15]
    seeds = [0, 1, 2]
    lambdas_sweep = [0.0, 0.02, 0.05, 0.1]

    for s in sizes:
        print("\n>>> Grid %sx%s | lambda sweep + paper figures <<<" % (s, s))
        base = {
            "w": s,
            "l": s,
            "n_modules": 0,
            "b_degrade": True,
            "per_degrade": 0.1,
            "max_electrode_uses": 50.0,
        }
        exp_lambda_sweep(
            base,
            lambdas=lambdas_sweep,
            seeds=seeds,
            n_e=1,
            n_s=64,
            num_iterations=50,
            num_steps=2000,
            log_root="log",
            paper_subdir="paper",
        )

    print("### Finished train.py successfully ###")
