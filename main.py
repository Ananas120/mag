import os
import logging
import subprocess
import tensorflow as tf

from datasets import set_dataset_dir, get_dataset
from experiments_mag import run_experiments
from utils import parse_args, get_object, set_display_options
from models import _models, get_pretrained
from models.model_utils import is_model_name, compare_models, remove_training_checkpoint
from loggers import TIME_LEVEL, add_handler, set_level


TOKEN = ''

_default_args = ['mode']

set_level(TIME_LEVEL)

def _get_config(* args, ** kwargs):
    args = [a for a in args if a not in kwargs]
    
    config = parse_args(* _default_args + args, add_unknown = True)
    for a in _default_args: config.pop(a)
    
    for k, v in config.items(): config.setdefault(k, v)
    
    logging.info('Config : {}'.format(config))
    
    return config

def get_strategy(multi_gpu, assert_gpu = False):
    gpus = tf.config.list_physical_devices('GPU')
    
    logging.info('tensorflow version : {}'.format(tf.__version__))
    logging.info('# of GPU : {}'.format(len(gpus)))
    
    if assert_gpu:
        assert len(gpus) > 0
    
    if multi_gpu == -1:
        return tf.distribute.MirroredStrategy() if multi_gpu and len(gpus) > 1 else tf.distribute.get_strategy()
    
    if not isinstance(multi_gpu, list): multi_gpu = [multi_gpu]
    tf.config.set_visible_devices([gpus[idx] for idx in multi_gpu], 'GPU')
    logging.info('# of visible GPU : {}'.format(len(tf.config.list_logical_devices('GPU'))))
    return tf.distribute.MirroredStrategy()

def build(** kwargs):
    config = _get_config('class', 'nom', multi_gpu = -1, ** kwargs)
    
    strategy = get_strategy(config.pop('multi_gpu'), assert_gpu = False)
    
    with strategy.scope():
        if 'pretrained_name' in config:
            if is_model_name(config['pretrained_name']):
                return _models.get(config.pop('class')).from_pretrained(** config)
            logging.warning('Pretrained model {} does not exists !'.format(config['pretrained_name']))

        return get_object(
            _models, config.pop('class'), err = True, print_name = 'Model class', ** config
        )

def train(** kwargs):
    config = _get_config('nom', 'dataset', multi_gpu = -1, dataset_dir = None, ** kwargs)
    strategy = get_strategy(config.pop('multi_gpu'), assert_gpu = True)
    
    ds_dir = config.pop('dataset_dir', None)
    if ds_dir: set_dataset_dir(ds_dir)
    dataset = get_dataset(config.pop('dataset'), ** config.pop('dataset_config', {}))
    
    with strategy.scope():
        model   = get_pretrained(config.pop('nom'))

        model.compile(** config.pop('compile_config', {}))

        if TOKEN: add_handler('telegram', token = TOKEN)

        hist = model.train(dataset, ** config)
    logging.info(hist)
    
    return hist

def test(** kwargs):
    config      = _get_config(
        'nom', 'dataset', multi_gpu = -1, dataset_dir = None, metrics = None, ** kwargs
    )
    strategy    = get_strategy(config.pop('multi_gpu'), assert_gpu = True)
    
    ds_dir      = config.pop('dataset_dir', None)
    if ds_dir: set_dataset_dir(ds_dir)
    dataset     = get_dataset(
        config.pop('dataset'), modes = ['valid'], ** config.pop('dataset_config', {})
    )['valid']
    

    with strategy.scope():
        model   = get_pretrained(config.pop('nom'))
        
        if TOKEN: add_handler('telegram', token = TOKEN)

        hist = model.test(dataset, ** config)
    logging.info(hist)
    
    return hist

def compare():
    pattern = _get_config(patterns = '').get('pattern', None)

    names = [n for n in os.listdir('pretrained_models') if not pattern or n.startswith(pattern)]

    infos = compare_models(names, True, True, epoch = 'best', add_training_config = True)
    
    set_display_options()
    print(infos.sort_values('val_loss'))

def experiments():
    if TOKEN: add_handler('telegram', token = TOKEN)

    run_experiments()

def clean_checkpoints():
    pattern = _get_config(patterns = '').get('pattern', None)

    names = [n for n in os.listdir('pretrained_models') if not pattern or n.startswith(pattern)]
    for n in names:
        remove_training_checkpoint(n)
    
def run_unitests():
    import test
    
    from unitest import run_tests
    
    config = parse_args(to_run = 'all', debug = True)
    
    run_tests(** config).assert_succeed()

_modes  = {
    'build' : build,
    'train' : train,
    'test'  : test,
    'compare'   : compare,
    'unitest'   : run_unitests,
    'clean'     : clean_checkpoints,
    'experiments'   : experiments
}

if __name__ == '__main__':
    mode   = parse_args(* _default_args)['mode']
    
    get_object(_modes, mode, err = True, print_name = 'Run mode')
