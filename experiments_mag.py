import os
import glob
import json
import logging
import subprocess
import tensorflow as tf

from utils import parse_args
from models.model_utils import _pretrained_models_folder, get_model_history, is_model_name

def simple_generator(model_name, bart_base = 'facebook/bart-large', ** kwargs):
    return {
        'class'             : 'AnswerGenerator',
        'lang'              : 'en',
        'input_format'      : ['{question}', '{context}'],
        'output_format'     : '{answer}',
        'text_encoder'      : bart_base,
        'max_input_length'  : 512,
    
        'pretrained' : bart_base,
        ** kwargs
    }

def simple_train_generator(model_name, retraining = False, ** kwargs):
    lr = 1e-5
    
    epochs = 1
    batch_size = 6
    return {
        'dataset'   : 'squad' if 'nq' not in model_name else 'nq',
        
        'compile_config'    : {
            'optimizer' : 'adam', 'optimizer_config' : {'lr' : lr}
        },

        'dataset_config'    : {
            'allow_la'          : False if 'osa' in model_name else True,
            'clean_text'        : True,
            'skip_impossible'   : True,
            'keep_only_first'   : True,
            'include_document'  : False,
            'shuffle'   : True
        },
        'epochs'    : epochs,
        'batch_size'    : batch_size,
        
        'shuffle_size'  : batch_size * 32,

        'max_input_length'  : 512,
        'max_output_length' : 32 * 3,
        ** kwargs
    }


def config_from_name(model_name, bart_base = 'facebook/bart-large', ** kwargs):
    if 'mag' not in model_name: return simple_generator(model_name, bart_base, ** kwargs)
    
    step, idx, mode = model_name.split('_')[-3 :]
    step, idx = int(step), int(idx)
    
    config = {
        'class'             : 'MAG',
        'nom'               : model_name,
        'lang'              : 'en',
        'output_format'     : '{answer}',
        'question_format'   : '{question}',
        'context_format'    : '{context}' if 'ct' not in model_name else '{title}{sep_token}{context}',
        'text_encoder'      : bart_base,
        'max_input_length'  : 512,
        'max_output_length' : 128,
        'context_offset'    : 128 if 'off' in model_name else -1,
        'subsample_question'    : False if 'entq' in model_name else False,

        'pretrained'    : bart_base,

        'encoder_repeat_pos_idx'    : True if 'rep' in model_name else False,
        'encoder_subsample_at'      : idx,
        'encoder_subsample_after'   : True if idx == 12 else False,
        'encoder_subsampling_step'  : step,
        'encoder_subsampling_offset': 0,
        'encoder_subsampling_mode'  : mode,

        'encoder_use_type_embedding': True if 'wt' in model_name else False,
        'encoder_max_types'         : 16,
        ** kwargs
    }
    
    if 'ft_doc' in model_name:
        config['pretrained_name'] = model_name.replace('ft_doc', 'ib')
    elif 'dense' in model_name: config['pretrained_name'] = model_name.replace('dense', 'mean')
    
    return config

def training_config_from_name(model_name, retraining = False, ** kwargs):
    if 'mag' not in model_name: return simple_train_generator(model_name, retraining, ** kwargs)
    
    step, idx, mode = model_name.split('_')[-3 :]
    step = int(step)
    
    lr = 1e-5
    if 'dense' in model_name:
        lr = {'name' : 'DivideByStep', 'maxval' : 1e-5, 'minval' : 1e-6, 'factor' : 0.1}

    use_doc = True if 'nq' in model_name and 'doc' in model_name else False
    
    epochs = 1 if 'dense' in model_name or retraining else max(1, step // 2 + 1)
    if 'nq' not in model_name:
        batch_size = 8
    else:
        if step < 2:
            batch_size = 3
        elif step == 2:
            batch_size = 4
        elif step == 3:
            batch_size = 5
        elif step == 5:
            batch_size = 6

    if use_doc: batch_size //= 2
    
    return {
        'dataset'   : 'squad' if 'nq' not in model_name else 'nq',
        
        'compile_config'    : {
            'optimizer' : 'adam', 'optimizer_config' : {'lr' : lr}
        },

        'dataset_config'    : {
            'allow_la'          : False if 'osa' in model_name else True,
            'clean_text'        : True,
            'skip_impossible'   : True,
            'keep_only_first'   : True,
            'include_document'  : use_doc,
            'shuffle'   : True
        },
        'is_rectangular'    : False if use_doc else True,
        
        'epochs'    : epochs,
        'batch_size'    : batch_size,
        
        'max_negatives' : 4,

        'shuffle_size'  : 0 if epochs == 0 else batch_size * 32,

        'augment_prct'  : 0. if use_doc else 0.25,
        'nb_mask'       : 1 if 'aug' not in model_name else 2,
        'min_mask_length'   : 1,
        'max_mask_length'   : 1 if 'aug' not in model_name else 2,

        'negative_mode'     : 'batch' if 'ib' in model_name else 'doc' if use_doc else None,

        'max_input_length'  : 512,
        'max_output_length' : 32 * 3,
        ** kwargs
    }

def testing_config_from_name(model_name, test_name, ** kwargs):
    use_doc = True if 'squad' not in test_name and 'doc' in test_name else False

    mode = 'none'
    if use_doc: mode = 'doc'
    elif 'ib' in test_name: mode = 'batch'
    config = {
        'dataset'   : 'squad' if 'squad' in test_name else 'nq',
        'test_name' : test_name,

        'dataset_config'    : {
            'allow_la'          : False if 'osa' in test_name else True,
            'clean_text'        : True,
            'skip_impossible'   : True,
            'keep_only_first'   : True,
            'include_document'  : use_doc,
            'shuffle'   : True
        },
        'is_rectangular'    : False if use_doc else True,
        
        'metrics'       : ['F1'],
        'add_loss'      : False,
        'batch_size'    : 12 if not use_doc else 4,
        'max_negatives' : 6,

        'negative_mode' : mode,
        'teacher_forcing_eval' : True if 'tf' in test_name else False,

        'max_input_length'     : 512,
        'max_output_length'    : 32 * 3,
        'run_eagerly'          : True,
        ** kwargs
    }
    
    if 'mag' not in model_name:
        config = {k : v for k, v in config.items() if 'negative' not in k}
    
    return config

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
    default_config.pop('mode')

    testing     = default_config.pop('test', False)
    test_name   = None if not testing else default_config.pop('test_name', 'test')
    
    names       = default_config.pop('names', names)
    allow_retraining    = default_config.pop('retrain', False)
    if not isinstance(names, (list, tuple)):
        names = [names] if '*' not in names else [
            os.path.basename(n) for n in glob.glob(os.path.join(_pretrained_models_folder, names))
        ]
    
    for name in names:
        if not testing:
            build_and_train(name, allow_retraining, ** default_config)
        else:
            test_model(name, test_name, ** default_config)

    
def test_model(name, test_name, ** default_config):
    hist = get_model_history(name)

    if hist is None:
        logging.warning('Model {} has not been trained yet, skip its test !'.format(name))
        return
    elif not is_model_name(name):
        logging.warning('Model {} does not exist, skip its test !'.format(name))
        return
    elif test_name + '_EM' in hist:
        logging.info('Test {} for {} already done !'.format(test_name, name))
        return
    
    config = config_to_list(testing_config_from_name(name, test_name, ** default_config))
    
    err = subprocess.run(['python3', 'main.py', 'test', name] + config)

    if err.returncode:
        logging.error('Error when testing model {}'.format(name))
        return

    logging.info('Successfully tested {} !'.format(name))
    
def build_and_train(name, allow_retraining, ** default_config):
    hist = get_model_history(name)

    retraining = False
    if hist is not None and len(hist) > 0:
        logging.info('Model {} has already been trained, {}'.format(
            name, "retraining it for 1 epoch" if allow_retraining else "skipping it."
        ))
        if not allow_retraining: return
        retraining = True
        
        
    if not is_model_name(name):
        config = config_to_list(config_from_name(name, ** default_config))
        
        err = subprocess.run(['python3', 'main.py', 'build'] + config)
    
        if err.returncode:
            logging.error('Error when building model {}'.format(name))
            return
        
    config = config_to_list(training_config_from_name(name, retraining, ** default_config))
    
    err = subprocess.run(['python3', 'main.py', 'train', name] + config)

    if err.returncode:
        logging.error('Error when training model {}'.format(name))
        return

    logging.info('Successfully built and trained {} !'.format(name))
