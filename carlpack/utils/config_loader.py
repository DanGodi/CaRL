# chimera/utils/config_loader.py
import yaml
from pathlib import Path

def load_configs(
    sim_cfg_path="configs/simulation_config.yaml",
    env_cfg_path="configs/env_config.yaml",
    train_cfg_path="configs/train_ppo_config.yaml") -> dict:
    """Loads all necessary YAML configuration files into a single dictionary."""
    
    configs = {}
    with open(sim_cfg_path, 'r') as f:
        configs['sim'] = yaml.safe_load(f)
    with open(env_cfg_path, 'r') as f:
        configs['env'] = yaml.safe_load(f)
    with open(train_cfg_path, 'r') as f:
        configs['train'] = yaml.safe_load(f)
        
    # Construct full paths from the base data_path
    data_path = Path(configs['sim']['data_path'])
    configs['sim']['target_data_path'] = str(data_path / configs['sim']['target_data_filename'])
    configs['sim']['model_save_path'] = str(data_path / configs['sim']['model_save_dirname'])
    configs['sim']['log_path'] = str(data_path / configs['sim']['log_dirname'])
    
    return configs