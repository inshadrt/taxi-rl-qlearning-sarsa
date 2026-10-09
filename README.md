# Passenger Pickup and Delivery with Q-learning and SARSA

Inshad Rahman Noman — PhDAI 830, University of the Cumberlands

This repository implements the Part 1 plan in Gymnasium 1.1.1 Taxi-v3. It includes real measured outputs, saved Q-tables, validation choices, repeated training runs, locked testing, and a native-environment demonstration. It is an educational simulation; the policies are not road-driving or dispatch systems.

## Quick start

Python 3.11–3.12 is recommended. From this folder:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m unittest discover -s tests -v
python run_project.py --smoke
```

The smoke run verifies the pipeline using 100 episodes and one final seed. Smoke outputs are separate and are NOT the results presented in the slide deck.

## Reproduce the full experiment

```bash
python run_project.py
python demo.py
```

The complete CPU experiment includes 24 development runs and 20 final training runs of 10,000 episodes each. Allow several minutes to tens of minutes depending on CPU speed. The demo requires FFmpeg on PATH, or use the provided MP4 directly. Set SDL_VIDEODRIVER=dummy for headless execution if needed. `analyze.py` rebuilds PNG plots and JSON summaries from CSV logs.

## Design

- 500 encoded states, six unmasked actions, one zero-initialized 500×6 Q-table per agent.
- Rewards: ordinary step −1, illegal pickup/drop-off −10, successful delivery +20.
- Discount 0.99; training budget 10,000 episodes; 200-step external time limit.
- Epsilon schedule: max(floor, 1 − (1 − floor) × episode / 8000).
- Q-learning bootstraps from max Q. SARSA uses the next epsilon-greedy action.
- True termination has zero bootstrap. External time-limit truncation retains bootstrap.
- No reward shaping, replay buffer, neural network, or action masking.

## Validation and testing

Development seeds: 0, 1, 2. Alpha: 0.1 or 0.3. Epsilon floor: 0.01 or 0.05. Validation reset seeds: 20000–20099. Rank settings by mean success, then mean return; ties prefer lower alpha then higher floor. The tie rule resolves an underspecified detail in Part 1 and was fixed before test evaluation.

Final training seeds: 100–109. The algorithms share each seed's per-episode reset list but have separate action RNGs. Greedy validation checkpoints occur at episode 0 and every 500 episodes. Final test reset seeds: 30000–30299. These are fresh draws from the SAME environment distribution, not held-out maps or guaranteed unseen states. Episode samples within a training run are not independent trained models.

Random baseline: uniformly sample all six actions. Planner baseline: breadth-first search through the known deterministic transition model to obtain a minimum-step delivery, including pickup and drop-off. Invalid or no-op transitions are excluded only in the planner, not the learners.

## Changes from the Part 1 plan

Training uses cached deterministic transition arrays derived directly from Gymnasium's `P` model for CPU efficiency. Agents receive only the sampled transition and reward in their update; the learning algorithms do not plan with the model. An exhaustive 3,000-transition test checks equivalence with `env.step`. Reset sampling uses actual Gymnasium. The demo uses actual Gymnasium `step` and `rgb_array` rendering.

Additional details: a validation tie rule, an initial checkpoint, and a 500-episode saved snapshot support reproducibility and learning diagnostics. The algorithms, environment version, rewards, episode budgets, final seeds, and test protocol stay consistent with Part 1.

## Outputs and interpretation

`outputs/selected_configs.json` records choices made ONLY with validation data.
`outputs/run_metrics.csv` holds one metric row per algorithm and training seed.
`outputs/test_episodes.csv` includes failures in return and success calculations.
`outputs/checkpoints.csv` records frozen greedy evaluation.
`outputs/training_episodes.csv` records exploratory behavior.
`outputs/summary.json` includes 95% percentile bootstrap intervals (10,000 resamples of training runs), standard deviations, and a paired Q-learning minus SARSA return difference. Timing excludes evaluation and plotting, but depends on the recorded machine. Conditional lengths and excess steps apply only to successful episodes. The learning threshold is the FIRST of three consecutive checkpoints at ≥95% success; it is a practical performance threshold, not proof of mathematical convergence.

Plots with faint individual curves display seed variability. A zero-width interval can occur when deterministic policies give identical final outcomes; it does not imply real-world certainty. Displayed demo seed 30001 and model seed 100 are illustrative, not evidence replacing the full test.

## File map

- `src/taxi_rl/core.py`: transitions, epsilon-greedy selection, updates, evaluation, planner.
- `run_project.py`: development sweep, frozen final training, locked testing.
- `analyze.py`: metrics, bootstrap intervals, learning curves.
- `demo.py`: native Gymnasium video and trajectory logs.
- `tests/test_core.py`: transition and numerical learning checks.
- `outputs/`: measured CSV/JSON data, figures, models, and demo video.
- `REFERENCES.md`: traceable research and documentation.
- `AI_DISCLOSURE.md`: assistance disclosure.

## Submission

Upload this complete folder to your own GitHub repository. Keep the requirements, source, README, saved results, and models. Open the repository link in a signed-out browser to verify accessibility, or grant your instructor access to a private repository. Submit that link together with the slides and your own 13–15 minute screen/webcam/voice recording. A downloadable archive does not automatically satisfy a requirement for a hosted repository URL.

## AI assistance and responsibility

OpenAI ChatGPT assisted with implementation, experiment execution, analysis, presentation organization, speaker notes, and documentation. Review the code and evidence, run at least the smoke check, and edit the narration to express your own understanding. AI assistance with substantive coding exceeds the assignment's expressly listed examples of brainstorming, organization, and clarity. Confirm that this level of assistance is permitted by your instructor before submitting. Do not represent the supplied draft as personally implemented without review.

## Verified Google Colab reproduction

The bundled outputs were reproduced by the presenter in Google Colab on October 9, 2026, using Python 3.13.16. All trained tables and all non-timing log values matched the original run. Mean update-loop times were 5.970 seconds for Q-learning and 5.494 seconds for SARSA; these exclude initialization, resets, evaluation and plotting. Timing is machine dependent. The supplied outputs and metadata are from that Colab reproduction.
