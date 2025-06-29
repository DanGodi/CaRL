# chimera/beamng_control/simulation_manager.py
from beamngpy import BeamNGpy, Scenario, Vehicle

class SimulationManager:
    """Manages the lifecycle of a BeamNG.tech simulation for the Chimera project."""
    
    def __init__(self, config: dict):
        self.config = config
        self.bng = None
        self.scenario = None
        self.base_vehicle = None
        self.target_vehicle = None # Optional ghost vehicle

    def connect(self):
        """Establishes connection with a running BeamNG instance."""
        self.bng = BeamNGpy(
            'localhost', 
            self.config['port'], 
            home=self.config['beamng_path'], 
            user=self.config['user_path']
        )
        self.bng.open()

    def setup_scenario(self, spawn_target=False):
        """
        Creates and loads the scenario with the base and (optionally) target vehicles.
        
        Args:
            spawn_target: If True, spawns a non-drivable target vehicle for visualization.
        """
        if not self.bng:
            raise ConnectionError("Must connect to BeamNG before setting up a scenario.")

        map_name = self.config['map']
        scenario_name = "chimera_scenario"
        self.scenario = Scenario(map_name, scenario_name)
        
        # Configure the base vehicle that our RL agent will control
        self.base_vehicle = Vehicle('base_car', 
                                    model=self.config['base_vehicle_model'],
                                    part_config=f"vehicles/{self.config['base_vehicle_model']}/{self.config['base_vehicle_config_name']}.pc")
        self.base_vehicle.props.color = 'Blue'
        self.scenario.add_vehicle(self.base_vehicle, pos=(-717, 101, 118), rot_quat=(0, 0, 0.38, 0.92))

        if spawn_target:
            self.target_vehicle = Vehicle('target_car', model=self.config['target_vehicle_model'])
            self.target_vehicle.props.color = 'Green'
            self.target_vehicle.ai.set_mode('disabled') # Make it a static object
            self.scenario.add_vehicle(self.target_vehicle, pos=(-717, 105, 118), rot_quat=(0, 0, 0.38, 0.92))

        self.scenario.make(self.bng)
        self.bng.load_scenario(self.scenario)
        print("Scenario loaded. Starting...")
        self.bng.start_scenario()
        
        # Crucial: Pause simulation immediately to allow for controlled steps
        self.bng.pause() 
        self.bng.step(5) # Step a few physics frames to let vehicles settle
        print("Simulation paused and ready.")

    def apply_vehicle_controls(self, params: dict):
        """
        Sets the active parameters on the base vehicle using the electrics system.
        This is the core control mechanism for the RL agent.

        Args:
            params: A dictionary mapping JBeam electrics keys to values.
                    e.g., {'front_spring_rate': 50000, 'rear_toe_angle': 0.5}
        """
        # I AM ASSUMING THE JBEAM 'electrics' SYSTEM IS SET UP CORRECTLY.
        # IF THIS DOES NOT WORK, THE JBEAM FILE IS THE PLACE TO DEBUG.
        # THE DICTIONARY KEYS HERE MUST MATCH THE `inputName` IN THE JBEAM `powertrain` SECTION.
        self.base_vehicle.electrics.update(params)
    
    def get_vehicle_state(self, vehicle: Vehicle) -> dict:
        """Polls vehicle sensors for a comprehensive state dictionary."""
        vehicle.sensors.poll()
        return vehicle.sensors
    
    def reset_vehicle_physics(self, vehicle: Vehicle):
        """Resets the physics state of a single vehicle."""
        self.bng.queue_lua_command(f'be:getObject("{vehicle.vid}"):reset()')
        
    def close(self):
        """Closes the connection to BeamNG."""
        if self.bng:
            print("Closing BeamNG connection...")
            self.bng.close()
            self.bng = None