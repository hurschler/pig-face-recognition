import logging.config
import yaml
import os

# Import and config logging
with open(os.path.join('../log_config.yaml'), 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)


