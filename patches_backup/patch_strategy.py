import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

config['patching']['type'] = 'regular'
config['patching']['regular'] = {'grid_size': 24}

with open("config.yaml", "w") as f:
    yaml.dump(config, f)
