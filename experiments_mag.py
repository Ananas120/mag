
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob
import json
import logging
import subprocess
import tensorflow as tf

from utils import parse_args
from models.model_utils import _pretrained_models_folder, get_model_history, get_models, is_model_name

PRED_DIR    = os.path.join('memoire_results', 'predictions')

def simple_generator(model_name, bart_base = 'facebook/bart-large', ** kwargs):
    return {
        'class'             : 'AnswerGenerator',
        'nom'               : model_name,
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
        'dataset'   : [
            ds_name for ds_name in ['nq', 'coqa', 'newsqa', 'squad'] if ds_name in model_name
        ],
        
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
    
    offset = -1
    if 'off64' in model_name:
        offset = 64
    elif 'off' in model_name:
        offset = 128
    
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
        
        'context_offset'    : offset,
        'split_contexts'    : True if 'split' in model_name else False,
        'subsample_question'    : False if 'entq' in model_name else False,

        'pretrained'    : bart_base,

        'encoder_repeat_pos_idx'    : True if 'rep' in model_name else False,
        'encoder_subsample_at'      : idx,
        'encoder_subsample_after'   : True if idx == 12 else False,
        'encoder_subsampling_step'  : step,
        'encoder_subsampling_offset': 0,
        'encoder_subsampling_mode'  : mode,

        'encoder_use_type_embedding': True if 'wt' in model_name else False,
        'encoder_max_types'         : 16 if 'split' not in model_name else 32,
        ** kwargs
    }
    
    if 'ft_doc' in model_name:
        config['pretrained_name'] = model_name.replace('ft_doc', 'ib')
    elif 'dense' in model_name: config['pretrained_name'] = model_name.replace('dense', 'mean')
    
    return config

def training_config_from_name(model_name, retraining = False, ** kwargs):
    if 'mag' not in model_name: return simple_train_generator(model_name, retraining, ** kwargs)
    
    datasets = model_name[3:].split('_mag')[0].split('_')

    step, idx, mode = model_name.split('_')[-3 :]
    step = int(step)
    
    lr = 1e-5
    if 'dense' in model_name:
        lr = {'name' : 'DivideByStep', 'maxval' : 1e-5, 'minval' : 1e-6, 'factor' : 0.1}

    use_doc = True if ('nq' in datasets and 'doc' in model_name) or 'qangaroo' in datasets else False
    
    if 'dense' in model_name or retraining:
        epochs = 1
    else:
        epochs = max(1, step // 2 + 1)
        #epochs += len(datasets) // 2

    if step < 2:
        batch_size = 3
    elif step == 2:
        batch_size = 4
    elif step == 3:
        batch_size = 5
    elif step > 3:
        batch_size = 6

    if use_doc: batch_size = max(1, batch_size // 2)
    if 'split' in model_name: batch_size = max(1, batch_size // 2)
    
    neg_mode = 'none'
    if use_doc: neg_mode = 'doc'
    elif 'ib' in model_name: neg_mode = 'batch'
    
    return {
        'dataset'   : datasets,
        
        'compile_config'    : {
            'optimizer' : 'adam', 'optimizer_config' : {'lr' : lr}
        },

        'dataset_config'    : {
            'keep_mode'         : 'longest' if 'osa' not in model_name else 'shortest',
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
        
        'max_negatives'     : 4 if 'split' not in model_name else 3,
        'max_sent_per_ctx'  : 5,

        'shuffle_size'  : 0 if epochs == 0 else batch_size * 32,

        'augment_prct'  : 0. if use_doc else 0.25,
        'nb_mask'       : 1 if 'aug' not in model_name else 2,
        'min_mask_length'   : 1,
        'max_mask_length'   : 1 if 'aug' not in model_name else 2,

        'negative_mode'     : neg_mode,

        'max_input_length'  : 512,
        'max_output_length' : 32 * 3,
        ** kwargs
    }

def testing_config_from_name(model_name, test_name, ** kwargs):
    if 'mag' not in model_name:
        step, idx, mode = 1, -1, None
    else:
        step, idx, mode = model_name.split('_')[-3 :]
        step, idx = int(step), int(idx)

    datasets = [
        ds_name for ds_name in ['squad', 'coqa', 'newsqa', 'qangaroo'] if ds_name in test_name
    ]
    if len(datasets) == 0 or 'nq' in test_name: datasets.append('nq')

    use_doc = True if 'doc' in test_name else False

    mode = 'none'
    if use_doc: mode = 'doc'
    elif 'ib' in test_name: mode = 'batch'
        
    batch_size = 12
    if use_doc and 'top5' in test_name: batch_size = 1
    elif use_doc: batch_size = 3
    elif 'top5' in test_name: batch_size = 6
    config = {
        'dataset'   : datasets,
        'test_name' : test_name,

        'dataset_config'    : {
            'keep_mode'         : 'all' if 'all' in test_name else 'longest',
            'allow_la'          : False if 'osa' in test_name else True,
            'clean_text'        : True,
            'skip_impossible'   : True,
            'keep_only_first'   : True,
            'include_document'  : use_doc,
            'shuffle'   : True
        },
        'is_rectangular'    : False if use_doc or 'all' in test_name else True,
        
        'metrics'       : ['F1'] if 'top5' not in test_name else ['TopKF1'],
        'add_loss'      : False,
        'batch_size'    : 1,
        
        'max_negatives'     : (5 * step - 1) if not 'split' in model_name else 5,
        'max_sent_per_ctx'  : 5,

        'negative_mode' : mode,
        'teacher_forcing_eval'  : True if 'tf' in test_name else False,
        'eval_infer_config'     : {} if 'top5' not in test_name else {'method' : 'beam'},
        
        'max_input_length'      : 512,
        'max_output_length'     : 32 * 3,
        ** kwargs
    }
    
    if 'mag' not in model_name:
        config = {k : v for k, v in config.items() if 'negative' not in k}
        config.pop('max_sent_per_ctx')
    
    return config

def predict_config_from_name(model_name, pred_name, ** kwargs):
    config = testing_config_from_name(
        model_name,
        pred_name,
        save        = True,
        directory   = os.path.join(PRED_DIR, model_name),
        filename    = '{}.json'.format(pred_name),
        ** kwargs
    )
    config.update({
        'is_rectangular'    : False,
        'metrics'       : ['f1'],
        'method'        : 'beam',
        'max_negatives' : 10
    })
    for k in ['add_loss', 'batch_size']:
        config.pop(k, None)
    
    config['dataset_config']['keep_mode'] = 'all'
    
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
    
    default_config = parse_args('mode', add_unknown = True)
    default_config.pop('mode')

    pred        = default_config.pop('pred', False)
    pred_name   = None if not pred else default_config.pop('pred_name', 'pred')

    testing     = default_config.pop('test', False)
    test_name   = None if not testing else default_config.pop('test_name', 'test')
    overwrite   = default_config.pop('overwrite', False)
    
    names       = default_config.pop('names', names)
    allow_retraining    = default_config.pop('retrain', False)
    if not isinstance(names, (list, tuple)):
        names = get_models(names) if '*' in names else [names]
    
    for name in names:
        success = build_and_train(name, allow_retraining, ** default_config)
        
        if testing and success:
            success = test_model(name, test_name, overwrite = overwrite, ** default_config)
        
        if pred and success:
            success = pred_model(name, pred_name, overwrite = overwrite, ** default_config)
        
        if not success:
            break

    
def build_and_train(name, allow_retraining, ** default_config):
    hist = get_model_history(name)

    retraining = False
    if hist is not None and len(hist) > 0:
        logging.info('Model {} has already been trained, {}'.format(
            name, "retraining it for 1 epoch" if allow_retraining else "skipping it."
        ))
        if not allow_retraining: return True
        retraining = True
        
        
    if not is_model_name(name):
        config = config_to_list(config_from_name(name, ** default_config))
        
        err = subprocess.run(['python3', 'main.py', 'build'] + config)
    
        if err.returncode:
            logging.error('Error when building model {}'.format(name))
            return True
        
    config = config_to_list(training_config_from_name(name, retraining, ** default_config))

    err = subprocess.run(['python3', 'main.py', 'train', name] + config)

    if err.returncode:
        logging.error('Error when training model {}'.format(name))
        return False

    logging.info('Successfully built and trained {} !'.format(name))
    return True

def test_model(name, test_name, overwrite = False, ** default_config):
    hist = get_model_history(name)

    suffix = '_EM'
    if 'top5' in test_name: suffix += '-1'
    if hist is None:
        logging.warning('Model {} has not been trained yet, skip its test !'.format(name))
        return True
    elif not is_model_name(name):
        logging.warning('Model {} does not exist, skip its test !'.format(name))
        return True
    elif hist.contains(test_name + suffix):
        if not overwrite:
            logging.info('Test {} for {} already done !'.format(test_name, name))
            return True
        
        logging.info('Overwriting test {}'.format(test_name))
        hist.pop(test_name)
        hist.save()
    
    config = config_to_list(testing_config_from_name(name, test_name, ** default_config))
    
    err = subprocess.run(['python3', 'main.py', 'test', name] + config)

    if err.returncode:
        logging.error('Error when testing model {}'.format(name))
        return False

    logging.info('Successfully tested {} !'.format(name))
    return True

def pred_model(name, pred_name, overwrite = False, ** default_config):
    hist = get_model_history(name)
    
    map_file    = os.path.join(PRED_DIR, name, pred_name + '.json')
    if hist is None:
        logging.warning('Model {} has not been trained yet, skip its prediction !'.format(name))
        return True
    elif not is_model_name(name):
        logging.warning('Model {} does not exist, skip its prediction !'.format(name))
        return True
    elif os.path.exists(map_file):
        if not overwrite:
            logging.info('Pred {} for {} already done !'.format(pred_name, name))
            return True
        
        logging.info('Overwriting prediction {}'.format(pred_name))
    
    config = config_to_list(predict_config_from_name(
        name, pred_name, overwrite = overwrite, ** default_config
    ))
    
    err = subprocess.run(['python3', 'main.py', 'predict', name] + config)

    if err.returncode:
        logging.error('Error when making prediction for model {}'.format(name))
        return False

    logging.info('Successfully predicted for {} !'.format(name))
    return True
