Of course. Here is the complete README content formatted as a single markdown block for easy copying.

# **Chimera - A Dynamic Vehicle Mimicry System**

This project trains a Reinforcement Learning agent to control a car's active systems (suspension, ballast, alignment) in the BeamNG.tech physics simulator. The goal is to make a "base" car dynamically mimic the driving characteristics of a "target" vehicle with high fidelity.

## **Project Overview**

The core idea is to create a "dynamic vehicle skin." The system takes a highly configurable base car and trains an AI policy to adjust its performance parameters in real-time. The policy's objective is to continuously modify these parameters so that the base car's telemetry data (G-forces, yaw rate, wheel speed, etc.) perfectly matches a pre-recorded telemetry trace from a target vehicle driving along the same path.

The entire system is developed and validated within the high-fidelity physics environment, **BeamNG.tech**, ensuring the learned driving dynamics are physically plausible.

## **How It Works: The Core Concept**

This project is a real-time control problem solved using Reinforcement Learning (RL).

*   **The Environment:** The BeamNG.tech simulation, containing our controllable base car.
*   **The Agent:** A **Proximal Policy Optimization (PPO)** neural network agent from the `stable-baselines3` library. The agent learns a policy that maps observations to actions.
*   **The Observation:** At each step, the agent observes the *error* between the target car's telemetry and the base car's current telemetry. It also sees the last action it took, giving it a sense of its current state.
*   **The Action:** The agent outputs a set of continuous values that correspond to settings for the active components, such as `front_spring_rate`, `rear_toe_angle`, and `ballast_position`.
*   **The Reward:** The agent is rewarded for minimizing the telemetry error. It is also given small penalties for making jerky, rapid changes to the controls or using extreme control inputs, encouraging smooth and efficient behavior.

The agent's goal is to learn a control policy that generalizes across different parts of a track, effectively learning the underlying physical "signature" of the target car.

## **System Requirements**

To run this project, you will need the following software installed:

*   **Python 3.9+**
*   **Git** for version control.
*   **BeamNG.tech (v0.27+ with a Research License)**. The research license is required for the Python API (`beamngpy`) to function.

## **Setup Instructions**

Follow these steps to set up the project environment.

**1. Clone the Repository**
```bash
git clone https://github.com/your-username/chimera_vehicle_mimicry.git
cd chimera_vehicle_mimicry
```

**2. Create a Python Virtual Environment**
It is highly recommended to use a virtual environment to manage dependencies.
```bash
python -m venv venv
```
Activate the environment. On Windows:
```cmd
.\venv\Scripts\activate
```
On macOS/Linux:
```bash
source venv/bin/activate
```

**3. Install Dependencies**
Install all required Python packages from the `requirements.txt` file.
```bash
pip install -r requirements.txt
```

**4. Configure Project Paths (Critical Step)**
You **MUST** edit the configuration file to tell the project where to find your BeamNG.tech installation and where to store data.

*   Open the file: `configs/simulation_config.yaml`
*   Modify the following paths to match your system:
    *   `beamng_path`: The absolute path to your BeamNG.tech installation folder.
    *   `user_path`: The absolute path to your BeamNG.tech user folder (often in `AppData/Local` or `Documents`).
    *   `data_path`: The path where all generated data, logs, and models will be stored. You can use a relative path like `"data"` to create a `data` folder inside the project directory, or an absolute path to another location.

**5. Configure the Vehicle JBeam (Critical Step)**
For the RL agent to control the car, the BeamNG physics engine must be told which vehicle parameters are "active." This is done with a custom JBeam part file.

*   An example file is provided at `vehicle_setups/etk800_chimera_base.jbeam`.
*   You must place this file (or your own modified version) into a BeamNG mod folder. A standard location is: `YourBeamNGUserFolder/mods/unpacked/your_mod_name/vehicles/etk800/`
*   This allows you to select the "Chimera Active Components" part in the vehicle customization menu in-game, which enables the real-time control hooks. Without this, the simulation will not work.

## **Usage Workflow**

Once set up, the project is run using the scripts in the `/scripts` directory. Follow these steps in order.

**Step 1: Generate Target Telemetry Data**
First, you need to create a ground-truth data file for the agent to mimic. This script records a target vehicle driving a path and saves its telemetry to a CSV file.

Launch BeamNG.tech and load into a map (e.g., Gridmap V2) with your chosen target vehicle. Then, run the following command in your terminal:
```bash
python scripts/01_generate_target_data.py
```
Follow the on-screen instructions to record the path. The output will be saved to the location specified by `data_path` in your config.

**Step 2: Train the Reinforcement Learning Agent**
This is the main training process. The script will launch BeamNG, create the environment, and start the PPO agent's training loop. This process can take a significant amount of time.

```bash
python scripts/02_train_agent.py
```
*   **Monitoring:** You can monitor the training progress in real-time using TensorBoard. Open a second terminal, activate the virtual environment, and run:
    ```bash
    tensorboard --logdir data/tensorboard_logs
    ```
    (Assuming your `data_path` is set to `"data"`).

**Step 3: Evaluate the Trained Model**
After training, you can test your model's performance. This script loads a saved model and runs it in the simulation, saving the resulting telemetry from the mimic car.

```bash
python scripts/03_evaluate_agent.py --model_path path/to/your/model.zip
```
Replace `path/to/your/model.zip` with the actual path to a model file saved during training (e.g., `data/training_runs/ppo_mimic_final.zip`).

**Step 4: Analyze and Plot the Results**
Finally, use the provided analysis script to generate plots comparing the target vehicle's telemetry against your agent's performance.

```bash
python chimera/analysis/plot_evaluation.py --target_csv [path_to_target.csv] --mimic_csv [path_to_mimic.csv] --output_dir [path_to_save_plots]
```
Example:
```bash
python chimera/analysis/plot_evaluation.py --target_csv data/target_telemetry/target_m3_gridmap_slalom.csv --mimic_csv data/evaluation_results/evaluation_results.csv --output_dir data/plots
```

## **Repository Structure**

*   **/configs/**: Contains all `.yaml` configuration files for experiments, the environment, and the simulation.
*   **/chimera/**: The core Python source code for the project, structured as a Python package.
*   **/scripts/**: High-level executable scripts that serve as the main entry points for running the project.
*   **/data/**: (Not tracked by Git) The default directory for all generated data, including target telemetry, training logs, model checkpoints, and evaluation results.
*   **/vehicle_setups/**: Contains example JBeam files demonstrating how to modify a vehicle for active control.
