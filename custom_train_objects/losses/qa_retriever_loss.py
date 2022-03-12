
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

class QARetrieverLoss(tf.keras.losses.Loss):
    def __init__(self, name = 'QARetrieverLoss', reduction = None, ** kwargs):
        super().__init__(name = name, reduction = 'none', ** kwargs)
    
    @property
    def metric_names(self):
        return ['loss', 'start_loss', 'end_loss']
    
    def call(self, y_true, y_pred):
        true_start, true_end = y_true
        pred_start, pred_end = y_pred
        
        start_loss  = tf.keras.losses.sparse_categorical_crossentropy(true_start, pred_start)
        end_loss    = tf.keras.losses.sparse_categorical_crossentropy(true_end, pred_end)
        
        return tf.stack([start_loss + end_loss, start_loss, end_loss], 0)
