# CaRL — Dynamic Vehicle Mimicry for BeamNG.tech

CaRL trains a reinforcement learning (RL) agent to make a configurable “base” car dynamically mimic a recorded “target” vehicle in BeamNG.tech. The agent observes the mismatch between the vehicles’ telemetry and adjusts active components (e.g., suspension factors) in real time to reduce the error.

- Core environment: [`carlpack.rl.mimic_env.MimicEnv`](carlpack/rl/mimic_env.py)
- Simulation orchestration: [`carlpack.beamng_control.simulation_manager.SimulationManager`](carlpack/beamng_control/simulation_manager.py)
- Telemetry streaming: [`carlpack.beamng_control.telemetry_streamer.TelemetryStreamer`](carlpack/beamng_control/telemetry_streamer.py)

Stable Baselines3 PPO is used for training and inference.

---

## Repository Structure

- [`configs/`](configs/)
  - [`simulation_config.yaml`](configs/simulation_config.yaml): BeamNG paths, vehicles, data/model locations.
  - [`env_config.yaml`](configs/env_config.yaml): Observation keys, action parameter mappings, reward weights.
  - [`train_ppo_config.yaml`](configs/train_ppo_config.yaml): PPO hyperparameters and training options.
- [`carlpack/`](carlpack/)
  - RL env: [`rl/mimic_env.py`](carlpack/rl/mimic_env.py)
  - BeamNG control: [`beamng_control/simulation_manager.py`](carlpack/beamng_control/simulation_manager.py)
  - Analysis: [`analysis/plot_rew.py`](carlpack/analysis/plot_rew.py)
  - Utils: [`utils/config_loader.py`](carlpack/utils/config_loader.py)
- [`scripts/`](scripts/)
  - Target data: [`01_generate_target_data.py`](scripts/01_generate_target_data.py)
  - Train PPO: [`02_train_agent.py`](scripts/02_train_agent.py)
  - Evaluate model: [`03_evaluate_agent.py`](scripts/03_evaluate_agent.py)
  - Resume training: [`04_resume_training.py`](scripts/04_resume_training.py)
  - Compare runs: [`05_show_improvement.py`](scripts/05_show_improvement.py)
- [`data/`](data/)
  - Example target: [`target_bolide.csv`](data/target_bolide.csv)
  - Results: [`evaluation_results/`](data/evaluation_results/), [`plots/`](data/plots/)
  - Training outputs: [`training_runs/`](data/training_runs/), [`tensorboard_logs/`](data/tensorboard_logs/)
- Root CSVs for convenience: [`training_reward.csv`](training_reward.csv), [`training_fps.csv`](training_fps.csv)

---

## Requirements

- Python 3.9+
- BeamNG.tech v0.27+ (Research license) and beamngpy
- Windows recommended (tested)
- GPU optional (PPO benefits)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Configuration

Edit the files in [`configs/`](configs/):

- [`simulation_config.yaml`](configs/simulation_config.yaml)
  - beamng_path, user_path
  - data_path (base folder for data, models, logs)
  - target_data_path (CSV with recorded target telemetry)
  - base_vehicle_model, target_vehicle_model, model_save_path
- [`env_config.yaml`](configs/env_config.yaml)
  - observation_keys: telemetry used by the agent
  - action_parameter_mappings: controllable parameters with min/max (e.g., carl_* factors)
  - reward_weights: shaping terms for tracking error and smoothness
- [`train_ppo_config.yaml`](configs/train_ppo_config.yaml)
  - PPO policy, learning rate, gamma, batch size, rollout length, etc.

The environment reads configs via [`carlpack.utils.config_loader.load_configs`](carlpack/utils/config_loader.py).

---

## How It Works

- Environment: [`MimicEnv`](carlpack/rl/mimic_env.py) steps the base vehicle along the target’s path, exposing:
  - Observations: error between target and base telemetry plus recent action.
  - Actions: continuous control values defined in [`env_config.yaml`](configs/env_config.yaml) under action_parameter_mappings.
  - Reward: negative tracking error with smoothness/penalty terms from reward_weights.
- Simulation: [`SimulationManager`](carlpack/beamng_control/simulation_manager.py) handles BeamNG launching, scenario setup, vehicle control, and telemetry polling.

---

## Workflow

1) Generate target telemetry (one-time per course/vehicle)
```bash
python scripts/01_generate_target_data.py
```
- Drives the target vehicle along a course and saves CSV to simulation_config.data_path (see target_data_path).

2) Train the PPO agent
```bash
python scripts/02_train_agent.py
```
- Uses [`MimicEnv`](carlpack/rl/mimic_env.py) and saves checkpoints/final model under simulation_config.model_save_path.
- Monitor with TensorBoard:
```bash
tensorboard --logdir data/tensorboard_logs
```

3) Evaluate a trained model
```bash
python scripts/03_evaluate_agent.py --model_path path/to/model.zip
```
- Runs a full course and writes telemetry to [`data/evaluation_results/`](data/evaluation_results/). See script flags for details.

4) Resume training from latest checkpoint
```bash
python scripts/04_resume_training.py
```
- Continues PPO training, preserving timestep counters and writing to the same run directory.

5) Show improvement: target → random-step → trained
```bash
python scripts/05_show_improvement.py
```
- Phase A: Target vehicle runs the recorded course.
- Phase B: Random-step baseline — base car samples a fresh random action each tick using ranges from [`env_config.yaml`](configs/env_config.yaml).
- Phase C: Trained PPO model — loads most recent model from simulation_config.model_save_path and runs the same course.

---

## Analysis

- FPS-independent reward plot: [`carlpack/analysis/plot_rew.py`](carlpack/analysis/plot_rew.py)
  - Merges [`training_reward.csv`](training_reward.csv) and [`training_fps.csv`](training_fps.csv) by step.
  - Produces reward × fps curves and a merged CSV.
  - Use `--help` for CLI options.

- Your evaluation CSVs from step 3 can be compared to the target reference for metrics/plots using your preferred tools.

---

## Tips and Troubleshooting

- BeamNG paths: Ensure beamng_path and user_path in [`simulation_config.yaml`](configs/simulation_config.yaml) are correct.
- Target CSV: Set target_data_path to a valid file (e.g., [`data/target_bolide.csv`](data/target_bolide.csv)).
- Models: If “No model files found” appears in [`05_show_improvement.py`](scripts/05_show_improvement.py), complete training or copy a model into model_save_path/checkpoints.
- Action ranges: The random-step baseline and PPO both rely on action_parameter_mappings from [`env_config.yaml`](configs/env_config.yaml). Keep min/max realistic.
- Headless vs GUI: BeamNG.tech may require GUI for recording target data; training can run minimized.

---

## Contributing

- Keep configs minimal and documented.
- Prefer adding new scripts under [`scripts/`](scripts/) with clear CLI flags.
- Reuse utilities in [`carlpack/`](carlpack/) and add tests where