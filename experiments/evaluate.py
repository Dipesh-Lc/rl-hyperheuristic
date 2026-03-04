import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

from collections import Counter
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import DQN

from rl_hh.env import SchedIdenticalHHEnv, EnvConfig
from rl_hh.baselines import run_random_hh, run_greedy_hh


def find_latest_model(results_dir: str = "results") -> str:
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Missing directory: {results_dir}")

    zips = [f for f in os.listdir(results_dir) if f.endswith(".zip") and f.startswith("dqn_sched_identical")]
    if not zips:
        raise FileNotFoundError("No trained model found in results/. Run python -m experiments.train_agent first.")

    zips.sort(key=lambda f: os.path.getmtime(os.path.join(results_dir, f)), reverse=True)
    return os.path.join(results_dir, zips[0])


def find_latest_models(results_dir: str = "results", k: int = 4) -> List[str]:
    """Return k newest models (oldest->newest), to match Stage1..Stagek ordering."""
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f"Missing directory: {results_dir}")

    zips = [f for f in os.listdir(results_dir) if f.endswith(".zip") and f.startswith("dqn_sched_identical")]
    if len(zips) < k:
        raise FileNotFoundError(f"Need at least {k} trained models in {results_dir}/ but found {len(zips)}.")

    zips.sort(key=lambda f: os.path.getmtime(os.path.join(results_dir, f)), reverse=True)
    newest = zips[:k]
    newest_paths = [os.path.join(results_dir, f) for f in newest][::-1]  # oldest->newest
    return newest_paths


def eval_rl(model: DQN, cfg: EnvConfig, episodes: int = 30):
    env = SchedIdenticalHHEnv(cfg)
    rows = []
    op_counts = Counter()

    for ep in range(episodes):
        obs, info = env.reset()
        lb = info["lb"]
        best = info["best_cmax"]
        instance_seed = info.get("instance_seed", None)
        start_ratio = info.get("ratio", None)

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            op_name = env.operators[int(action)].name
            op_counts[op_name] += 1
            obs, r, terminated, truncated, info = env.step(int(action))
            best = min(best, info["best_cmax"])
            done = terminated or truncated

        rows.append(
            {
                "episode": ep,
                "instance_seed": instance_seed,
                "start_ratio": start_ratio,
                "best_cmax": best,
                "lb": lb,
                "best_ratio": best / (lb + 1e-12),
            }
        )

    env.close()
    return rows, op_counts


def summarize(df: pd.DataFrame, method: str, tag: str, n: int, m: int, dist: str) -> dict:
    return {
        "tag": tag,
        "n": n,
        "m": m,
        "dist": dist,
        "method": method,
        "episodes": len(df),
        "mean_best_ratio": float(df["best_ratio"].mean()),
        "median_best_ratio": float(df["best_ratio"].median()),
        "std_best_ratio": float(df["best_ratio"].std()),
    }


def run_one_size(model: DQN, cfg: EnvConfig, episodes: int, results_dir: str, tag: str, rl_label: str):
    """Run RL + Random + Greedy for one cfg and return dataframes + op_counts."""
    # RL
    rl_rows, op_counts = eval_rl(model, cfg, episodes=episodes)
    df_rl = pd.DataFrame(rl_rows)

    print(f"\n[{tag}] {rl_label} operator counts:", op_counts)
    pd.DataFrame(op_counts.items(), columns=["operator", "count"]).to_csv(
        os.path.join(results_dir, f"op_counts_{tag}_{rl_label}.csv"),
        index=False,
    )

    # RandomHH 
    env_rnd = SchedIdenticalHHEnv(cfg)
    df_rnd = pd.DataFrame(run_random_hh(env_rnd, episodes=episodes))

    # GreedyHH
    env_grd = SchedIdenticalHHEnv(cfg)
    df_grd = pd.DataFrame(run_greedy_hh(env_grd, episodes=episodes))

    return df_rl, df_rnd, df_grd


if __name__ == "__main__":
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # evaluation grid 
    sizes: List[Tuple[str, int, int, int]] = [
        ("n50_m5", 50, 5, 200),
        ("n120_m8", 120, 8, 300),
        ("n250_m15", 250, 15, 400),
        ("n500_m25", 500, 25, 600),
    ]

    dist = "bimodal"
    episodes = 30
    init_solution = "random"
    step_penalty = 0.0001
    seed = 123


    # FIGURE A: Latest model evaluated across ALL sizes (2x2)

    latest_path = find_latest_model(results_dir)
    print("Evaluating latest model (generalization):", latest_path)
    latest_model = DQN.load(latest_path)

    all_summaries_A = []
    figA, axesA = plt.subplots(2, 2, figsize=(12, 8))
    axesA = axesA.flatten()

    for idx, (tag, n, m, max_steps) in enumerate(sizes):
        cfg = EnvConfig(
            n=n, m=m, dist=dist, seed=seed,
            max_steps=max_steps, step_penalty=step_penalty,
            init_solution=init_solution,
        )

        df_rl, df_rnd, df_grd = run_one_size(
            latest_model, cfg, episodes, results_dir, tag, rl_label="LatestRL"
        )

        # Save raw results 
        df_rl.to_csv(os.path.join(results_dir, f"eval_{tag}_latest_rl.csv"), index=False)
        df_rnd.to_csv(os.path.join(results_dir, f"eval_{tag}_random.csv"), index=False)
        df_grd.to_csv(os.path.join(results_dir, f"eval_{tag}_greedy.csv"), index=False)

        all_summaries_A.append(summarize(df_rl, "RL(Latest)", tag, n, m, dist))
        all_summaries_A.append(summarize(df_rnd, "RandomHH", tag, n, m, dist))
        all_summaries_A.append(summarize(df_grd, "GreedyHH", tag, n, m, dist))

        ax = axesA[idx]
        ax.boxplot([df_rl["best_ratio"], df_rnd["best_ratio"], df_grd["best_ratio"]])
        ax.set_xticklabels(["RL", "Random", "Greedy"])
        ax.set_title(f"{tag} ({dist})")
        ax.set_ylabel("Best ratio (makespan / LB)")
        ax.grid(False)

    summaryA = pd.DataFrame(all_summaries_A)
    summaryA.to_csv(os.path.join(results_dir, "eval_summary_latest_generalization.csv"), index=False)

    figA.suptitle(f"Latest RL: Generalization Across Sizes (init={init_solution}, episodes={episodes})")
    figA.tight_layout()
    figA.savefig(os.path.join(results_dir, "eval_latest_generalization_2x2.png"), dpi=200)

   
    # FIGURE B: Stage-matched evaluation (diagonal): Stage i on size i

    stage_paths = find_latest_models(results_dir, k=4)
    stage_models = {
        "Stage1": DQN.load(stage_paths[0]),
        "Stage2": DQN.load(stage_paths[1]),
        "Stage3": DQN.load(stage_paths[2]),
        "Stage4": DQN.load(stage_paths[3]),
    }

    print("\nStage models (oldest->newest):")
    for s, p in zip(["Stage1", "Stage2", "Stage3", "Stage4"], stage_paths):
        print(f" - {s}: {p}")

    all_summaries_B = []
    figB, axesB = plt.subplots(2, 2, figsize=(12, 8))
    axesB = axesB.flatten()

    for idx, (stage_name, (tag, n, m, max_steps)) in enumerate(zip(["Stage1", "Stage2", "Stage3", "Stage4"], sizes)):
        cfg = EnvConfig(
            n=n, m=m, dist=dist, seed=seed,
            max_steps=max_steps, step_penalty=step_penalty,
            init_solution=init_solution,
        )

        df_rl, df_rnd, df_grd = run_one_size(
            stage_models[stage_name], cfg, episodes, results_dir, tag, rl_label=stage_name
        )

        # Save raw results per stage-size
        df_rl.to_csv(os.path.join(results_dir, f"eval_{tag}_{stage_name}_rl.csv"), index=False)
        df_rnd.to_csv(os.path.join(results_dir, f"eval_{tag}_{stage_name}_random.csv"), index=False)
        df_grd.to_csv(os.path.join(results_dir, f"eval_{tag}_{stage_name}_greedy.csv"), index=False)

        all_summaries_B.append(summarize(df_rl, f"RL({stage_name})", tag, n, m, dist))
        all_summaries_B.append(summarize(df_rnd, "RandomHH", tag, n, m, dist))
        all_summaries_B.append(summarize(df_grd, "GreedyHH", tag, n, m, dist))

        ax = axesB[idx]
        ax.boxplot(
            [df_rl["best_ratio"], df_rnd["best_ratio"], df_grd["best_ratio"]])
        ax.set_xticklabels(["RL", "Random", "Greedy"])
        ax.set_title(f"{stage_name}: {tag} ({dist})")
        ax.set_ylabel("Best ratio (makespan / LB)")
        ax.grid(False)

    summaryB = pd.DataFrame(all_summaries_B)
    summaryB.to_csv(os.path.join(results_dir, "eval_summary_stage_matched.csv"), index=False)

    figB.suptitle(f"Stage-Matched RL (diagonal): Stage i evaluated on its training size (episodes={episodes})")
    figB.tight_layout()
    figB.savefig(os.path.join(results_dir, "eval_stage_matched_2x2.png"), dpi=200)

    print("\nSaved:")
    print(" - results/eval_latest_generalization_2x2.png")
    print(" - results/eval_summary_latest_generalization.csv")
    print(" - results/eval_stage_matched_2x2.png")
    print(" - results/eval_summary_stage_matched.csv")