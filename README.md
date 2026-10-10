# Passenger Pickup and Delivery with Q-learning and SARSA

Inshad Rahman Noman — PhDAI 830, University of the Cumberlands

For this project, I compared Q-learning and SARSA in the Taxi-v3 environment using Gymnasium 1.1.1. My goal was to train an agent to pick up a passenger and deliver them to the correct destination while avoiding illegal actions and unnecessary steps.

I followed the plan from Part 1 and included the source code, trained Q-tables, evaluation results, graphs, and an agent demonstration in this repository. This project is an educational simulation, so its results do not establish readiness for real-world driving or dispatch.

## Getting started

Python 3.11–3.12 is recommended for running the project locally. Open a terminal in the project folder and run:

    python -m venv .venv

Activate the environment on macOS or Linux:

    source .venv/bin/activate

On Windows PowerShell, use:

    .venv\Scripts\Activate.ps1

Then install the dependencies, run the tests, and try a short smoke run:

    pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python run_project.py --smoke

The smoke run uses 100 episodes and one final training seed to check the pipeline. Its outputs are saved separately and are not the results used in my presentation.

## Running the full experiment

To reproduce the complete experiment and generate the demonstration, run:

    python run_project.py
    python demo.py

The experiment includes 24 development runs and 20 final training runs, with 10,000 episodes per run. Runtime depends on the computer.

Generating the demo requires FFmpeg on PATH. The completed MP4 is also included in the outputs folder. For a headless environment, set `SDL_VIDEODRIVER=dummy` if needed.

The `analyze.py` script rebuilds the graphs and summary files from the CSV logs.

## How I designed the agents

I used a separate, zero-initialized 500 × 6 Q-table for each agent. Taxi-v3 has 500 encoded states and six actions. Both agents could select all six actions, including illegal pickup and drop-off actions.

I kept the environment’s original rewards:

- −1 for an ordinary step.
- −10 for an illegal pickup or drop-off.
- +20 for a successful delivery.

I used a discount factor of 0.99, a training budget of 10,000 episodes, and a maximum of 200 steps per episode.

Exploration decreased according to:

    epsilon = max(floor, 1 − (1 − floor) × episode / 8000)

The main difference between the algorithms is their update target. Q-learning uses the highest next-state Q-value, while SARSA uses the next action selected by its epsilon-greedy policy.

For a true terminal state, I used zero bootstrap. When an episode ended because of the external time limit, I retained the bootstrap value. I did not use reward shaping, neural networks, replay buffers, or action masking.

## Training, validation, and testing

I first compared four parameter settings for each algorithm using development seeds 0, 1, and 2. I tested learning rates of 0.1 and 0.3 and exploration floors of 0.01 and 0.05.

I evaluated these settings using validation reset seeds 20000–20099. Settings were ranked by mean success rate, followed by mean return. For ties, I preferred the lower learning rate and then the higher exploration floor. I fixed this rule before testing.

For final training, I used seeds 100–109, giving ten trained agents per algorithm. Each paired run used the same episode reset list, with separate random generators for action selection.

I evaluated frozen greedy policies before training and every 500 episodes. Final testing used reset seeds 30000–30299. These were fresh samples from the same environment distribution; they were not new maps or guaranteed unseen states.

I also included two baselines:

- A random agent that selected uniformly from all six actions.
- A breadth-first-search planner that used the known transition model to find a minimum-step delivery route.

Only the planner excluded invalid or no-op transitions. The learning agents received no such restriction.

## Changes from my Part 1 plan

To make training more efficient on a CPU, I used cached deterministic transitions taken directly from Gymnasium’s transition model. The agents still learned from individual transitions and rewards; they did not use the model to plan routes.

I checked all 3,000 state-action transitions against the actual environment’s `step` function. Reset sampling used Gymnasium, and the demonstration used the actual environment’s stepping and image rendering.

I also added the validation tie rule, an initial evaluation checkpoint, and a saved model after 500 episodes. These details helped make the experiment reproducible and show how the agent’s behavior changed during learning.

The algorithms, environment version, rewards, training budget, final seeds, and testing protocol remained consistent with my Part 1 plan.

## Understanding the outputs

The outputs folder contains the results used in my analysis:

- `selected_configs.json`: settings selected using validation data.
- `run_metrics.csv`: metrics for each algorithm and training seed.
- `test_episodes.csv`: individual test results, including failures.
- `checkpoints.csv`: frozen greedy evaluations during training.
- `training_episodes.csv`: results from exploratory training episodes.
- `summary.json`: summary statistics, bootstrap intervals, and the paired return difference between Q-learning and SARSA.

I calculated 95% percentile bootstrap intervals using 10,000 resamples of the trained runs. I treated each trained run as the unit of comparison rather than treating individual episodes as independent models.

For my learning-speed comparison, I used the first of three consecutive checkpoints with at least 95% success. This is a practical performance threshold, not proof of mathematical convergence.

Episode lengths and excess steps reported conditionally apply only to successful episodes. Timing measures the update loop and depends on the computer used.

The faint lines in the learning plots show variation across training seeds. Some final intervals have zero width because the deterministic policies produced identical outcomes. That does not establish certainty outside this simulation.

The demo uses reset seed 30001 and model seed 100 to illustrate behavior. My conclusions are based on the full evaluation, not just that example.

## Repository files

- `src/taxi_rl/core.py`: transitions, action selection, learning updates, evaluation, and planner.
- `run_project.py`: parameter comparison, final training, and testing.
- `analyze.py`: statistics, bootstrap intervals, and graphs.
- `demo.py`: demonstration video and trajectory logs.
- `tests/test_core.py`: transition and numerical learning checks.
- `outputs/`: results, graphs, trained models, and demo video.
- `REFERENCES.md`: research sources and documentation.
- `AI_DISCLOSURE.md`: explanation of AI assistance.

## AI assistance and responsibility

I used OpenAI ChatGPT to help me understand reinforcement learning concepts and assist with code, analysis, and presentation preparation. I ran the project in Google Colab and reviewed the outputs. I am responsible for checking my final submission and understanding the methods and results.

## My Google Colab reproduction

I reproduced the project outputs in Google Colab on October 9, 2026, using Python 3.13.16. Verification showed that all trained Q-tables and all non-timing log values matched the original run.

The mean update-loop time was 5.970 seconds for Q-learning and 5.494 seconds for SARSA. These measurements exclude initialization, resets, evaluation, and plotting, so they do not represent the total runtime.

The bundled outputs and metadata come from my Colab reproduction. Runtime may differ on another computer.
