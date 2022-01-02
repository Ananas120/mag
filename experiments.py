import json
import logging
import subprocess
import tensorflow as tf

from utils import parse_args
from models.model_utils import get_model_history, is_model_name

def config_from_name(model_name, bart_base = 'facebook/bart-large'):
    raise NotImplementedError

def training_config_from_name(model_name):
    raise NotImplementedError

def config_to_list(config):
    config_list = []
    for k, v in config.items():
        config_list.append('--{}'.format(k))
        if not isinstance(v, (list, tuple)): v = [v]
        config_list.extend([json.dumps(vi) if not isinstance(vi, str) else vi for vi in v])
    
    return config_list

def run_experiments(names = [], ** kwargs):
    logging.info('tensorflow version : {}\n# GPU : {}'.format(
        tf.__version__, len(tf.config.list_physical_devices('GPU'))
    ))
    tf.config.set_visible_devices([], 'GPU')
    
    default_config = parse_args('mode', add_unknown = True, multi_gpu = -1, dataset_dir = None)

    names = default_config.pop('names', names)
    default_config.pop('mode')
    allow_retraining = default_config.pop('retrain', False)
    if not isinstance(names, (list, tuple)): names = [names]
    
    for name in names:
        hist = get_model_history(name)

        retraining = False
        if hist is not None and len(hist) > 0:
            logging.info('Model {} has already been trained, {}'.format(
                name, "retraining it for 1 epoch" if allow_retraining else "skipping it."
            ))
            if not allow_retraining: continue
            retraining = True
        
        
        if not is_model_name(name):
            config = config_to_list(config_from_name(name, ** default_config))
            
            err = subprocess.run(['python3', 'main.py', 'build'] + config)
        
            if err.returncode:
                logging.error('Error when building model {}'.format(name))
                continue
        
        config = config_to_list(training_config_from_name(name, retraining, ** default_config))
        
        err = subprocess.run(['python3', 'main.py', 'train', name] + config)

        if err.returncode:
            logging.error('Error when training model {}'.format(name))
            continue

        logging.info('Successfully built and trained {} !'.format(name))

        