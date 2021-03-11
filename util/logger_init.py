import logging.config
import yaml
import os

# Import and config logging
main_base = os.path.dirname(__file__)
log_config_file = os.path.join(main_base, 'log_config.yaml')

with open(log_config_file, 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)


