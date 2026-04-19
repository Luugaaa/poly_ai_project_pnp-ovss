import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

config['postprocessing']['crf']['pos_w'] = 7
config['postprocessing']['crf']['bi_w'] = 10

with open("config.yaml", "w") as f:
    yaml.dump(config, f)

