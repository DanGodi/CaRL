# CaRL

This project trains a Reinforcement Learning agent to control a car's active systems (suspension, ballast, alignment) in BeamNG.tech to make it dynamically mimic the driving characteristics of a target vehicle.

## Project Goal

The system takes a "base" car with controllable parameters and trains a policy that adjusts these parameters in real-time to match the telemetry of a "target" car (e.g., a BMW M3) driving along the same path.

## Core Workflow

This project is designed for a dual-boot (macOS/Windows) setup with a shared data drive.

1.  **Develop (macOS):** Write and edit Python code in the `chimera/` package. Define experiment parameters in the `configs/` directory. Use Git for version control.
2.  **Sync Code:** `git push` from macOS, `git pull` on Windows.
3.  **Prepare Data (Windows):** Run `scripts/01_generate_target_data.py` to create a reference telemetry file for the target vehicle. This is a one-time step for each new track/car combination.
4.  **Train (Windows):** Run the main training script: `python scripts/02_train_agent.py`. Monitor progress using TensorBoard pointing to the log directory on the shared drive.
5.  **Evaluate (Windows):** Test a trained model's performance: `python scripts/03_evaluate_agent.py`. This generates a CSV of the mimic car's telemetry.
6.  **Analyze (macOS/Windows):** Use `chimera/analysis/plot_evaluation.py` or Jupyter Notebooks to compare the target and mimic telemetry data from the shared drive. Iterate on designs and configurations.

## System Setup

1.  **Dual-Boot & Shared Drive:**
    *   Ensure an Intel Mac has a Windows partition via Bootcamp.
    *   Create a shared drive or partition formatted as **ExFAT**. This drive will hold the `data/` directory.
    *   Update `data_path` in `configs/simulation_config.yaml` to point to this location (e.g., `E:/chimera_project/data` on Windows, `/Volumes/Shared/chimera_project/data` on macOS).

2.  **Software Installation (on BOTH macOS and Windows):**
    *   Install Python 3.9+.
    *   Clone this repository: `git clone <your_repo_url>`
    *   Create a virtual environment: `python -m venv venv` and `source venv/bin/activate` (or `venv\Scripts\activate` on Windows).
    *   Install dependencies: `pip install -r requirements.txt`.

3.  **BeamNG.tech (on Windows):**
    *   Install BeamNG.tech (research license).
    *   Update the `beamng_path` and `user_path` in `configs/simulation_config.yaml` to match your installation.

## Vehicle Setup

The base vehicle (e.g., `etk800`) must be modified to accept real-time inputs. This is done by adding custom variables to the vehicle's JBeam file that are controlled by the `electrics` system. See `vehicle_setups/etk800_chimera_base.jbeam` for an example.