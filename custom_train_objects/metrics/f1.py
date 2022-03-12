
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

import tensorflow as tf

class F1(tf.keras.metrics.Metric):
    def __init__(self, decode_fn = None, normalize = True, exclude = None, name = 'F1',
                 ** kwargs):
        super().__init__(name = name)
        self.normalize  = normalize
        self.exclude    = exclude
        self.decode_fn  = decode_fn
        
        self.samples    = self.add_weight("batches", initializer = "zeros", dtype = tf.int32)
        
        self.exact_match    = self.add_weight("exact_match",    initializer = "zeros")
        self.precision      = self.add_weight("precision",      initializer = "zeros")
        self.recall         = self.add_weight("recall",         initializer = "zeros")
        self.f1             = self.add_weight("f1",             initializer = "zeros")
    
    @property
    def metric_names(self):
        return ["EM", "F1", "precision", "recall"]
    
    def decode(self, data, skip_empty = False):
        if self.decode_fn is not None:
            decoded = self.decode_fn(data)
            if isinstance(decoded[0], list) and skip_empty:
                last_valid_idx = [
                    max([i+1 for i, d in enumerate(dec) if len(d) > 0] + [1])
                    for dec in decoded
                ]
                decoded = [
                    dec[:last_valid] for dec, last_valid in zip(decoded, last_valid_idx)
                ]
            return decoded
        return data.numpy()
    
    def compute_f1(self, y_true, y_pred):
        from utils.text import f1_score
        
        return f1_score(
            self.decode(y_true, skip_empty = True), self.decode(y_pred, skip_empty = False),
            normalize = self.normalize, exclude = self.exclude
        )
    
    def tf_compute_f1(self, y_true, y_pred):
        results = tf.py_function(
            self.compute_f1, [y_true, y_pred], Tout = tf.float32
        )
        results = tf.reshape(results, [tf.shape(y_true)[0], -1, 4])
        
        bests   = tf.expand_dims(tf.argmax(results[:,:,1], axis = 1), axis = 1)
        results = tf.gather(results, bests[:,0], axis = 1, batch_dims = 1)

        results = tf.reduce_sum(results, axis = 0)
        
        return results
    
    def update_state(self, y_true, y_pred, sample_weight = None):
        if isinstance(y_true, (list, tuple)): y_true = y_true[0]
        
        if len(tf.shape(y_true)) == 1: y_true = tf.expand_dims(y_true, 0)
        if len(tf.shape(y_pred)) == 1: y_pred = tf.expand_dims(y_pred, 0)
        if y_pred.dtype == tf.float32:
            if len(tf.shape(y_pred)) > 2: y_pred = tf.cast(tf.argmax(y_pred, axis = -1), tf.int32)
            else: y_pred = tf.cast(y_pred > 0.5, tf.int32)
        
        scores = self.tf_compute_f1(y_true, y_pred)
        
        self.samples.assign_add(tf.shape(y_true)[0])
        
        self.exact_match.assign_add(scores[0])
        self.f1.assign_add(scores[1])
        self.precision.assign_add(scores[2])
        self.recall.assign_add(scores[3])
    
    def result(self):
        n = tf.cast(self.samples, tf.float32)
        return self.exact_match / n, self.f1 / n, self.precision / n, self.recall / n
    
    def get_config(self):
        config = super().get_config()
        config['normalize']  = self.normalize
        config['exclude']    = self.exclude
        return config
