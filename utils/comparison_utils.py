
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
import numpy as np
import pandas as pd
import tensorflow as tf

def is_in(target, value, nested_test = False, ** kwargs):
    if nested_test:
        cmp         = [is_in(target, v, ** kwargs) for v in value]
        invalids    = [(i, msg) for i, (eq, msg) in enumerate(cmp) if not eq]
        if len(invalids) == 0: return True, ''
        
        return False, "Invalid items ({}) :\n{}".format(
            len(invalids), '\n'.join(['Item #{} : {}'.format(i, msg) for i, msg in invalids])
        )
    try:
        if not isinstance(target, (list, tuple)): target = [target]
        missing = [k for k in target if k not in value]
        return len(missing) == 0, '{} are missing ({})'.format(missing, value)
    except TypeError as e:
        return False, str(e)

def is_smaller(target, value):
    try:
        return value < target, 'value is{} smaller than target'.format('' if value < target else ' not')
    except TypeError as e:
        return False, str(e)

def is_greater(target, value):
    try:
        return value > target, 'value is{} greater than target'.format('' if value < target else ' not')
    except TypeError as e:
        return False, str(e)

def is_equal(target, value, ** kwargs):
    try:
        compare(target, value, ** kwargs)
        return True, 'Value are equals !'
    except AssertionError as e:
        return False, str(e)

def is_diff(target, value, ** kwargs):
    eq, msg = is_equal(target, value, ** kwargs)
    return not eq, msg
    
def compare(target, value, ** kwargs):
    for t, compare_fn in _comparisons.items():
        if isinstance(target, t):
            compare_fn(target, value, ** kwargs)
            return

    if hasattr(target, 'get_config'):
        compare(target.get_config(), value.get_config())
        return
    
    compare_primitive(target, value, ** kwargs)

def compare_types(value, allowed_types, ** kwargs):
    assert isinstance(value, allowed_types), "Value of type {} is not in valid types {}\n  Value : {}".format(
        type(value), allowed_types, value
    )
    
def compare_primitive(target, value, ** kwargs):
    assert target == value, "'{}' != '{}'".format(target, value, target == value)

def compare_str(target, value, raw_compare_if_filename = False, ** kwargs):
    try:
        from models.model_utils import is_model_name
    except ImportError:
        is_model_name = lambda n: False

    if raw_compare_if_filename or len(target) >= 512:
        compare_primitive(target, value, ** kwargs)
    elif os.path.isfile(target):
        if not isinstance(value, str):
            compare(_load_file(target), value, ** kwargs)
        else:
            compare_file(target, value, ** kwargs)
    elif is_model_name(target):
        compare_base_model(target, value, ** kwargs)
    else:
        compare_primitive(target, value, ** kwargs)

def compare_list(target, value, nested_test = False, ** kwargs):
    if nested_test: target = [target] * len(value)
    assert len(target) == len(value), "Target length {} != value length {}".format(len(target), len(value))
    
    try:
        if target == value: return
    except ValueError as e:
        pass
    
    cmp         = [is_equal(it1, it2, ** kwargs) for it1, it2 in zip(target, value)]
    invalids    = [(i, msg) for i, (eq, msg) in enumerate(cmp) if not eq]

    assert len(invalids) == 0, "Invalid items ({}) :\n{}".format(
        len(invalids), '\n'.join(['Item #{} : {}'.format(i, msg) for i, msg in invalids])
    )
    
def compare_dict(target, value, fields = None, ** kwargs):
    if fields is not None:
        target = {k : target[k] for k in target if k in fields}
        value = {k : value[k] for k in value if k in fields}

    missing_v_keys  = [k for k in target if k not in value]
    missing_t_keys  = [k for k in value if k not in target]
        
    assert len(missing_v_keys) + len(missing_t_keys) == 0, "Missing keys in value : {}\nAdditionnal keys in value : {}".format(missing_v_keys, missing_t_keys)
    
    cmp         = {k : is_equal(target[k], value[k]) for k in target}
    invalids    = {k : msg for k, (eq, msg) in cmp.items() if not eq}
    
    assert len(invalids) == 0, "Invalid items ({}) :\n{}".format(
        len(invalids), '\n'.join(['Key {} : {}'.format(k, msg) for k, msg in invalids.items()])
    )

def compare_array(target, value, max_err = 1e-5, err_mode = 'abs', ** kwargs):
    if isinstance(target, tf.Tensor): target = target.numpy()
    if not isinstance(value, np.ndarray): value = np.array(value)
    assert target.shape == value.shape, "Target shape {} != value shape {}".format(target.shape, value.shape)
    
    assert target.dtype == value.dtype, "Target dtype {} != value dtype {}".format(target.dtype, value.dtype)
    
    if target.dtype in (np.bool, np.object):
        assert np.all(target == value), "Vallue differ for target with dtype {}".format(target.dtype)
    else:
        err = np.abs(target - value)
        
        if err_mode == 'norm':
            err = err / (np.abs(target) + 1e-6)

        assert np.all(err <= max_err), "Values differ : max {} - mean {} - min {}".format(
            np.max(err), np.mean(err), np.min(err)
        )

def compare_dataframe(target, value, ignore_index = True, ** kwargs):
    missing_v_cols  = [k for k in target if k not in value]
    missing_t_cols  = [k for k in value if k not in target]
    
    assert len(missing_v_cols) + len(missing_t_cols) == 0, "Missing keys in value : {}\nAdditionnal keys in value : {}".format(missing_v_cols, missing_t_cols)
    
    assert len(target) == len(value), "Target length {} != value length {}".format(len(target), len(value))
    
    if not ignore_index:
        diff = np.where(~np.all((target == value).values, axis = -1))[0]
        assert len(diff) == 0, "DataFrames differs ({})\n  Target : \n{}\n  Value : \n{}".format(
            len(diff), target.iloc[diff], value.iloc[diff]
        )
        return
    
    invalids = []
    for idx, row in value.iterrows():
        if not np.any(np.all((row == target).values, axis = -1)):
            invalids.append(idx)
    
    assert len(invalids) == 0, "Some columns are not in target ({}) :\n{}".format(
        len(invalids), value.iloc[invalids]
    )

def _load_file(filename):
    assert os.path.exists(filename), "Filename {} does not exist !".format(filename)
    
    global _file_loading_fn
    if _file_loading_fn is None:
        from utils.file_utils import _load_file_fn
        from utils.audio import _audio_formats, read_audio
        from utils.image import _image_formats, load_image

        _file_loading_fn    = {
            ** {audio_ext : read_audio for audio_ext in _audio_formats},
            ** {image_ext : load_image for image_ext in _image_formats},
            ** _load_file_fn
        }

    ext = os.path.splitext(filename)[1][1:]

    assert ext in _file_loading_fn, "Extension {} unhandled, cannot load data from file {}".format(ext, filename)
    
    return _file_loading_fn[ext](filename)

def compare_file(target, value, ** kwargs):
    assert os.path.exists(target), "Target file {} does not exist !".format(target)
    assert os.path.exists(value), "Value file {} does not exist !".format(value)
    
    t_ext = os.path.splitext(target)[1][1:]
    v_ext = os.path.splitext(value)[1][1:]
    
    assert t_ext == v_ext, "Extensions differ {} vs {}".format(target, value)
    
    t_data = _load_file(target)
    v_data = _load_file(value)
    
    eq, msg = is_equal(t_data, v_data, raw_compare_if_filename = True)
    
    assert eq, 'Data of files {} and {} differ : {}'.format(target, value, msg)

def compare_base_model(target, value, ** kwargs):
    from models.model_utils import is_model_name, get_model_infos
    
    assert is_model_name(target), "Target {} is not a valid model !".format(target)
    assert is_model_name(value), "Value {} is not a valid model name !".format(value)
    
    t_infos = get_model_infos(target)
    v_infos = get_model_infos(value)
    
    compare(t_infos, v_infos)

_comparisons    = {
    str     : compare_str,
    (list, tuple)   : compare_list,
    (dict, pd.Series)   : compare_dict,
    (np.ndarray, tf.Tensor) : compare_array,
    pd.DataFrame    : compare_dataframe
}

_file_loading_fn    = None