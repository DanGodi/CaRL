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
        
        bng = BeamNGpy('localhost', self.config['port'], home=self.config['beamng_path'])
        try:
            # The launch=True flag tells open() to start the simulator
            bng.open(launch=True)
            self.bng = bng
            print("Successfully launched and connected to BeamNG.tech.")
        except Exception as e:
            print(f"FATAL: Failed to launch. Is the 'beamng_path' in your config correct?")
            raise e
    
    def setup_scenario(self, vehicle_model, vehicle_config=None, spawn_target=False):
        """
        Creates and loads a scenario with a primary vehicle.
        """
        if not self.bng:
            raise ConnectionError("Cannot setup scenario. Must call launch() first.")

        map_name = self.config['map']
        scenario_name = "carl_scenario"
        
        self.scenario = Scenario(map_name, scenario_name, description="CaRL scenario")
        
        # Create the primary vehicle for this scenario
        # We now pass the model and config as arguments
        self.base_vehicle = Vehicle('base_car', 
                                    model=vehicle_model, license='CaRL', 
                                    color='Blue',
                                    part_config=vehicle_config) # Use the provided config

        self.base_vehicle.color = 'Blue'
        self.scenario.add_vehicle(self.base_vehicle, pos=(5.660,-15,100.928), rot_quat=(0, 0, 1, 0))

        # Finalize and load the scenario
        self.scenario.make(self.bng)
        self.bng.load_scenario(self.scenario)
        print(f"Scenario '{scenario_name}' loaded on map '{map_name}'.")
        self.bng.start_scenario()
        
        self.bng.pause() 
        self.bng.step(5)
        print("Simulation paused and ready.")

    def apply_vehicle_controls(self, params: dict):
        """Sets the active parameters on the base vehicle using the electrics system."""
        for key, value in params.items():
            # Create a command like: "electrics.values.carl_spring_factor = 0.75"
            lua_command = f"electrics.values.{key} = {value}"
            self.base_vehicle.queue_lua_command(lua_command)

    def reset_vehicle_physics(self, vehicle: Vehicle):
        """Resets the physics state of a single vehicle."""
        #reset a vehicle without reloading the whole scenario
        self.bng.teleport_vehicle(vehicle.vid, vehicle.state['pos'], vehicle.state['rot'])
        
    def close(self):
        """Closes the connection to BeamNG."""
        if self.bng:
            print("Closing BeamNG connection...")
            self.bng.close()
            self.bng = None