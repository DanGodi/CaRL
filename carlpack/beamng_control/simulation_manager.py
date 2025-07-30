# carl/beamng_control/simulation_manager.py
# --- FULLY CORRECTED VERSION ---

from beamngpy import BeamNGpy, Scenario, Vehicle

class SimulationManager:
    """Manages the lifecycle of a BeamNG.tech simulation for the CaRL project."""
    
    def __init__(self, config: dict):
        # The config is the 'sim' section of your YAML config files
        self.config = config
        self.bng = None
        self.scenario = None
        self.base_vehicle = None
        self.target_vehicle = None

    def launch(self):
        """
        Launches a NEW instance of BeamNG.tech and connects to it.
        This is used for automated training and evaluation.
        """
        print("Attempting to launch a new BeamNG.tech instance...")
        # The 'home' path points to the installation directory and is
        # required to tell beamngpy where the .exe file is.
        bng = BeamNGpy('localhost', self.config['port'], home=self.config['beamng_path'])
        try:
            # The launch=True flag tells open() to start the simulator
            bng.open(launch=True)
            self.bng = bng
            print("Successfully launched and connected to BeamNG.tech.")
        except Exception as e:
            print(f"FATAL: Failed to launch. Is the 'beamng_path' in your config correct?")
            raise e
            
    def setup_scenario(self, spawn_target=False):
        """
        Creates and loads the scenario with the base and (optionally) target vehicles.
        """
        if not self.bng:
            raise ConnectionError("Cannot setup scenario. Must call launch() or connect() first.")

        map_name = self.config['map']
        scenario_name = "carl_scenario"
        
        # Create a new scenario object
        self.scenario = Scenario(map_name, scenario_name, description="CaRL training scenario")
        
        # Create the base vehicle that our RL agent will control
        # The name 'base_car' is a temporary vehicle ID used by beamngpy
        self.base_vehicle = Vehicle('base_car', 
                                    model=self.config['base_vehicle_model'],
                                    part_config=f"vehicles/{self.config['base_vehicle_model']}/{self.config['base_vehicle_config_name']}.pc")
        self.base_vehicle.color = 'Blue'
        self.scenario.add_vehicle(self.base_vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.38, 0.92))

        # Optionally spawn a second vehicle to represent the target visually
        if spawn_target:
            self.target_vehicle = Vehicle('target_car', model=self.config['target_vehicle_model'])
            self.target_vehicle.color = 'Green'
            self.target_vehicle.ai.set_mode('disabled')
            self.scenario.add_vehicle(self.target_vehicle, pos=(-717, 105, 118), rot_quat=(0, 0, 0.38, 0.92))

        # Finalize and load the scenario into the simulator
        self.scenario.make(self.bng)
        self.bng.load_scenario(self.scenario)
        print(f"Scenario '{scenario_name}' loaded on map '{map_name}'.")
        self.bng.start_scenario()
        
        # Pause the simulation immediately to allow for controlled steps
        self.bng.pause() 
        self.bng.step(5) # Step a few physics frames to let vehicles settle
        print("Simulation paused and ready.")

    def apply_vehicle_controls(self, params: dict):
        """Sets the active parameters on the base vehicle using the electrics system."""
        self.base_vehicle.electrics.update(params)
    
    def reset_vehicle_physics(self, vehicle: Vehicle):
        """Resets the physics state of a single vehicle."""
        # This is a more modern way to reset a vehicle without reloading the whole scenario
        self.bng.teleport_vehicle(vehicle.vid, vehicle.state['pos'], vehicle.state['rot'])
        
    def close(self):
        """Closes the connection to BeamNG."""
        if self.bng:
            print("Closing BeamNG connection...")
            self.bng.close()
            self.bng = None