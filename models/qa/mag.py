
# Copyright (C) 2022 Langlois Quentin. All rights reserved.
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

import custom_architectures.transformers_arch.mag_arch as mag_arch

from loggers import timer
from models.qa.base_generator import BaseGenerator
from models.qa.answer_generator_split import AnswerGeneratorSplit

class MAG(AnswerGeneratorSplit):
    def _build_model(self, pretrained, ** kwargs):
        super(BaseGenerator, self)._build_model(
            model = mag_arch.MAG.from_pretrained(
                pretrained, return_attention = False, ** kwargs
            )
        )

    @property
    def subsampling_factor(self):
        return self.model.hparams.encoder_subsampling_step

    def __str__(self):
        des = super().__str__()
        des += "- # of embedding layers : {}\n".format(len(self.encoder.embedding_layers))
        des += "- # of memory layers : {}\n".format(len(self.encoder.memory_layers))
        des += "- Subsampling factor : {}\n".format(self.subsampling_factor)
        des += "- Subsampling mode : {}\n".format(self.model.hparams.encoder_subsampling_mode)
        return des
    
    @timer(name = 'prediction', log_if_root = False)
    def call(self,
             inputs,
             training               = False,
             merge_contexts         = False,
             force_not_subsampling  = False,
             ** kwargs
            ):
        n_contexts  = len(inputs) // 2 - 2
        
        q_not_subsampling = force_not_subsampling if self.subsample_question else ([True] + [force_not_subsampling] * n_contexts)
        positional_offset = -1 if self.context_offset <= 0 else ([-1] + [self.context_offset] * n_contexts)
        
        return self.model(
            inputs,
            encoder_positional_offset   = positional_offset,
            merge_contexts          = merge_contexts or self.force_merging or (training and self.in_batch_negatives),
            force_not_subsampling   = q_not_subsampling,
            training                = training,
            return_attention        = False,
            return_hidden_states    = False,
            return_mask             = False,
            ** kwargs
        )
    
    @timer(name = 'inference', log_if_root = False)
    def infer(self,
              inputs,
              training              = False,
              merge_contexts        = False,
              force_not_subsampling = False,
              ** kwargs
             ):
        kwargs.setdefault('max_length', self.max_output_length)
        
        n_contexts  = len(inputs) // 2 - 1
        
        q_not_subsampling = force_not_subsampling if self.subsample_question else ([True] + [force_not_subsampling] * n_contexts)
        positional_offset = -1 if self.context_offset <= 0 else ([-1] + [self.context_offset] * n_contexts)
        
        return self.model.infer(
            inputs,
            encoder_positional_offset   = positional_offset,
            merge_contexts          = merge_contexts or self.force_merging or (training and self.in_batch_negatives),
            force_not_subsampling   = q_not_subsampling,
            training                = training,
            return_attention        = False,
            return_hidden_states    = False,
            return_mask             = False,
            ** kwargs
        )
