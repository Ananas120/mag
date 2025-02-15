
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

import logging
import numpy as np
import pandas as pd
import tensorflow as tf

MAX_MATRIX_SIZE = 512 * 1024 * 1024

def distance(x, y, method, as_matrix = False, max_matrix_size = MAX_MATRIX_SIZE, ** kwargs):
    """
        Compute distance between `x` and `y` with `method` function
        
        Arguments : 
            - x : 1D vector or 2D matrix
            - y : 2D matrix of points
            - method : string (the name of the method)
            - as_matrix : whether to compute matrix or point-wise distance (see notes)
            - max_matrix_size : maximum number of values in the matrix distance computation
            - kwargs : kwargs to pass to the distance function
        Return : 
            - `method` function applied to `x` and `y`
        
        Note : 
        If `as_matrix is True` : return a matrix such that `matrix[i, j]` is the distance between `x[i]` and `y[j]`
        Else : `matrix[i]` is the distance between `x[i]` and `y[i]`
        
        This distance can be a scalar (euclidian, manhattan, dot-product) or a vector of element-wise distance (l1, l2)
        
        Important note : this function returns a **distance** score it means that the lower the score, the more similar they are ! If the method is a similarity metric (such as dot-product), the function returns the inverse (- distances) to keep this property
    """
    distance_fn = method if callable(method) else _distance_methods.get(method, None)
    if distance_fn is None:
        raise ValueError("Distance method is not callable or does not exist !\n  Accepted : {}\n  Got : {}".format(
            list(_distance_methods.keys()), method
        ))
    
    if method in _str_distance_method:
        return _str_distance_method[method](x, y, ** kwargs)

    if len(tf.shape(x)) == 1: x = tf.expand_dims(x, axis = 0)
    if len(tf.shape(y)) == 1: y = tf.expand_dims(y, axis = 0)
    
    if as_matrix:
        if len(tf.shape(x)) == 2: x = tf.expand_dims(x, axis = 1)
        if len(tf.shape(y)) == 2: y = tf.expand_dims(y, 0)
    elif len(tf.shape(x)) == 2 and len(tf.shape(y)) == 3:
        x = tf.expand_dims(x, axis = 1)

    max_x, max_y = -1, -1
    if max_matrix_size > 0:
        max_x = tf.minimum(max_matrix_size // tf.shape(x)[-1] + 1, tf.shape(x)[0])
        max_y = tf.minimum(max_matrix_size // (max_x * tf.shape(x)[-1]) + 1, tf.shape(y)[1])
        
    
    if max_x != -1 and (max_x < tf.shape(x)[0] or max_y < tf.shape(y)[1]):
        distances = []
        for i in range(0, tf.shape(x)[0], max_x):
            distances.append(tf.concat([
                distance_fn(x[i : i + max_x], y[:, j : j + max_y], ** kwargs)
                for j in range(0, tf.shape(y)[1], max_y)
            ], axis = -1))
        distances = tf.concat(distances, axis = 0)
    else:
        distances = distance_fn(x, y, ** kwargs)
        
    return distances if method != 'dp' else -distances # dot_product is a similarity metric

def dot_product(x, y, ** kwargs):
    return tf.squeeze(tf.matmul(x, y, transpose_b = True), axis = 1)

def l1_distance(x, y, ** kwargs):
    return tf.abs(x - y)

def l2_distance(x, y, ** kwargs):
    return tf.square(x, y)

def manhattan_distance(x, y, ** kwargs):
    return tf.reduce_sum(tf.abs(x - y), axis = -1)

def euclidian_distance(x, y, ** kwargs):
    return tf.math.sqrt(tf.reduce_sum(tf.square(x - y), axis = -1))

def edit_distance(hypothesis,
                  truth,
                  partial   = False,
                  deletion_cost     = {},
                  insertion_cost    = {}, 
                  replacement_cost  = {},
                  normalize     = True,
                  return_matrix = False,
                  verbose   = False,
                  ** kwargs
                 ):
    """
        Compute a weighted Levenstein distance
        
        Arguments :
            - hypothesis    : the predicted value   (iterable)
            - truth         : the true value        (iterable)
            - partial       : whether to make partial alignment or not
            - insertion_cost    : weights to insert a new symbol
            - replacement_cost  : weights to replace a symbol (a --> b) but 
            is not in both sens (a --> b != b --> a) so you have to specify weights in both sens
            - normalize     : whether to normalize on truth length or not
            - return_matrix : whether to return the matrix or not
            - verbose       : whether to show costs for path or not
        Return :
            - distance if not return_matrix else (distance, matrix)
                - distance  : scalar, the Levenstein distance between `hypothesis` and truth `truth`
                - matrix    : np.ndarray of shape (N, M) where N is the length of truth and M the length of hypothesis. 
        
        Note : if `partial` is True, the distance is the minimal distance
        Note 2 : `distance` (without normalization) corresponds to the "number of errors" between `hypothesis` and `truth`. It means that the start of the best alignment (if partial) is `np.argmin(matrix[-1, 1:]) - len(truth) - distance`
    """
    matrix = np.zeros((len(hypothesis) + 1, len(truth) + 1))
    # Deletion cost
    deletion_costs = np.array([0] + [deletion_cost.get(h, 1) for h in hypothesis])
    matrix[:, 0] = np.cumsum(deletion_costs)
    # Insertion cost
    if not partial:
        matrix[0, :] = np.cumsum([0] + [insertion_cost.get(t, 1) for t in truth])

    truth_array = truth if not isinstance(truth, str) else np.array(list(truth))
    for i in range(1, len(hypothesis) + 1):
        deletions = matrix[i-1, 1:] + deletion_costs[i]
        
        matches   = np.array([replacement_cost.get(hypothesis[i-1], {}).get(t, 1) for t in truth])
        matches   = matrix[i-1, :-1] + matches * (truth_array != hypothesis[i-1])
        
        min_costs = np.minimum(deletions, matches)
        for j in range(1, len(truth) + 1):
            insertion   = matrix[i, j-1] + insertion_cost.get(truth[j-1], 1)

            matrix[i, j] = min(min_costs[j-1], insertion)
    
    if verbose:
        columns = [''] + [str(v) for v in truth]
        index = [''] + [str(v) for v in hypothesis]
        logging.info(pd.DataFrame(matrix, columns = columns, index = index))
    
    distance = matrix[-1, -1] if not partial else np.min(matrix[-1, 1:])
    if normalize:
        distance = distance / len(truth) if not partial else distance / len(hypothesis)
    
    return distance if not return_matrix else (distance, matrix)

def hamming_distance(hypothesis, truth, replacement_matrix = {}, normalize = True,
                     ** kwargs):
    """
        Compute a weighted hamming distance
        
        Arguments : 
            - hypothesis    : the predicted value   (iterable)
            - truth         : the true value        (iterable)
            - replacement_matrix    : weights to replace element 1 to 2 (from hypothesis to truth). Note that this is not in 2 sens so a --> b != b --> a
            - normalize     : whether to normalize on truth length or not
        Return : distance between hypothesis and truth (-1 if they have different length)
    """
    if len(hypothesis) != len(truth): return -1
    distance = sum([
        replacement_matrix.get(c1, {}).get(c2, 1)
        for c1, c2 in zip(hypothesis, truth) if c1 != c2
    ])
    if normalize: distance = distance / len(truth)
    return distance


_str_distance_method    = {
    'hamming'   : hamming_distance,
    'edit'      : edit_distance
}

_distance_methods = {
    ** _str_distance_method,
    'dp'        : dot_product,
    'l1'        : l1_distance,
    'l2'        : l2_distance,
    'manhattan' : manhattan_distance,
    'euclidian' : euclidian_distance
}