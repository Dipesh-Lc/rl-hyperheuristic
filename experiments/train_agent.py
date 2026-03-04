from rl_hh.rl import train_dqn

if __name__ == "__main__":

    # Stage 1
    path1 = train_dqn(
        out_dir="results",
        seed=0,
        n=50, m=5,
        dist="bimodal",
        max_steps=200,
        total_timesteps=600_000,
        init_solution="random",
        step_penalty=0.00005,
        load_path=None,
    )

    # Stage 2
    path2 = train_dqn(
        out_dir="results",
        seed=0,
        n=120, m=8,
        dist="bimodal",
        max_steps=300,
        total_timesteps=800_000,
        init_solution="random",
        step_penalty=0.00005,
        load_path=path1,
    )

    # Stage 3
    path3 = train_dqn(
        out_dir="results",
        seed=0,
        n=250, m=15,
        dist="bimodal",
        max_steps=400,
        total_timesteps=1_000_000,
        init_solution="random",
        step_penalty=0.00005,
        load_path=path2,
    )

    # Stage 4
    path4 = train_dqn(
        out_dir="results",
        seed=0,
        n=500, m=25,
        dist="bimodal",
        max_steps=600,
        total_timesteps=1_200_000,
        init_solution="random",
        step_penalty=0.00005,
        load_path=path3,
    )

    print("Final model:", path4)