
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

from custom_train_objects.metrics.f1 import F1

def cum_argmax(x, axis = -1):
    def cond(i, x, cum_max):
        return i <= tf.shape(x)[-1]
    
    def body(i, x, cum_max):
        max_i = tf .expand_dims(tf.argmax(x[..., :i], axis = -1, output_type = tf.int32), axis = -1)
        return i + 1, x, tf.concat([cum_max, max_i], axis = -1)
    
    return tf.while_loop(
        cond = cond, body = body, loop_vars = (2, x, tf.fill(tf.shape(x[..., :1]), 0))
    )[2]

class TopKF1(F1):
    def __init__(self, k = 5, decode_fn = None, normalize = True, exclude = None,
                 name = 'F1', ** kwargs):
        super(F1, self).__init__(name = name)
        self.k      = k
        self.normalize  = normalize
        self.exclude    = exclude
        self.decode_fn  = decode_fn
        
        self.samples    = self.add_weight("batches", initializer = "zeros", dtype = tf.int32)
        
        self.exact_match    = self.add_weight("exact_match",    initializer = "zeros", shape = (k, ))
        self.precision      = self.add_weight("precision",      initializer = "zeros", shape = (k, ))
        self.recall         = self.add_weight("recall",         initializer = "zeros", shape = (k, ))
        self.f1             = self.add_weight("f1",             initializer = "zeros", shape = (k, ))
    
    @property
    def metric_names(self):
        _names = ["EM", "F1", "precision", "recall"]
        
        metrics = []
        for k in range(1, self.k + 1):
            metrics.extend(['{}-{}'.format(n, k) for n in _names])
        return metrics
    
    def tf_compute_f1(self, y_true, y_pred):
        assert len(tf.shape(y_pred) == 3)
        y_pred = y_pred[:, :self.k]
        results = tf.py_function(
            self.compute_f1, [y_true, y_pred], Tout = tf.float32
        )
        # shape is [batch_size, n_true, n_pred, scores]
        results.set_shape([None, None, None, 4])

        results = tf.transpose(results, [0, 2, 1, 3])
        
        bests   = tf.argmax(results[..., 1], axis = -1)
        results = tf.gather(results, bests, axis = 2, batch_dims = 2)
        
        cum_best    = cum_argmax(results[..., 1])
        results     = tf.gather(results, cum_best, batch_dims = 1, axis = 1)
        
        results = tf.reduce_sum(results, axis = 0)
        results = tf.transpose(results)
        #tf.print("Results :(", tf.shape(results), ")\n", results)

        return results
    
    def reset_state(self):
        for var in self.variables:
            var.assign(tf.zeros(shape = var.shape, dtype = var.dtype))
    
    def result(self):
        n = tf.cast(self.samples, tf.float32)
        result = tf.stack([
            self.exact_match / n, self.f1 / n, self.precision / n, self.recall / n
        ])
        return tf.reshape(tf.transpose(result), [-1])
    
    def get_config(self):
        config = super().get_config()
        config['k'] = self.k
        return config
