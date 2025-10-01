from myDataset import CSVDataset
from trainer import Trainer
from mlp import model_ff
import argparse
import importlib.util
import sys
from pathlib import Path

def load_config(config_path):
    config_path = Path(config_path).resolve()
    module_name = config_path.stem  # filename without .py

    spec = importlib.util.spec_from_file_location(module_name, config_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


# read the config file
parser = argparse.ArgumentParser(description="Run training with a given config file")
parser.add_argument("--config", type=str, required=True, help="Path to the Python config file (e.g., configs/config_dev.py)")
args = parser.parse_args()

# Load the config
config = load_config(args.config)
trainer_config = config.trainer
model_config = config.model
general_config = config.general

# Create a model
model = model_ff(input_size = model_config['input_size'])

# Create a trainer
tconfig = Trainer.get_default_config()
tconfig.merge_from_dict(trainer_config)
train_dataset = CSVDataset(csv_file=general_config['train_table_path'], 
                           feature_cols = general_config['feature_cols'], 
                           target_col = general_config['target_col'])
#test_dataset = CSVDataset(general_config['test_table_path'])
trainer = Trainer(config = tconfig, 
                  model=model, 
                  train_dataset=train_dataset)