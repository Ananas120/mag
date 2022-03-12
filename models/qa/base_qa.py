
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

import os
import pandas as pd
import tensorflow as tf

from models.base_model import BaseModel
from utils.text import get_encoder, random_mask
from models.weights_converter import partial_transfer_learning

DEFAULT_MAX_INPUT_LENGTH = 512

def find_index(text, answer, start_idx = 0):
    idx = -1
    possible_starts = tf.where(text == answer[0])
    if len(tf.shape(possible_starts)) == 2:
        for i in tf.cast(tf.squeeze(possible_starts, axis = 1), tf.int32):
            tokens = text[i : i + len(answer)]
            if len(tokens) == len(answer) and tf.reduce_all(tokens == answer):
                idx = i
                break

    return idx

class BaseQAModel(BaseModel):
    def __init__(self,
                 lang,
                 
                 input_format   = ['{question}', '{context}'],
                 output_format  = '{answer}',
                 
                 text_encoder   = None,
                 max_input_length   = DEFAULT_MAX_INPUT_LENGTH,
                 use_fixed_length_input = False,
                 
                 pretrained = None,
                 
                 ** kwargs
                ):
        if use_fixed_length_input: raise NotImplementedError()
        
        self.lang   = lang
        self.input_format   = input_format
        self.output_format  = output_format
        self.max_input_length   = max_input_length
        self.use_fixed_length_input = use_fixed_length_input
        
        # Initialization of Text Encoder
        self.text_encoder = get_encoder(text_encoder = text_encoder, lang = lang)
        
        kwargs.setdefault('pretrained_name', pretrained)
        super().__init__(pretrained = pretrained, ** kwargs)
                
        # Saving text encoder and mel fn (if needed)
        if not os.path.exists(self.text_encoder_file):
            self.text_encoder.save_to_file(self.text_encoder_file)
        
        if hasattr(self.model, '_build'): self.model._build()
    
    def init_train_config(self,
                          max_input_length = None,
                          
                          nb_mask   = 1,
                          min_mask_length   = 1,
                          max_mask_length   = 1,
                          
                          ** kwargs
                         ):
        if max_input_length: self.max_input_length   = max_input_length
        
        self.nb_mask = nb_mask
        self.min_mask_length    = min_mask_length
        self.max_mask_length    = max_mask_length
        
        super().init_train_config(** kwargs)
    
    @property
    def training_hparams(self):
        return super().training_hparams(
            max_input_length = None,
            nb_mask   = 1,
            min_mask_length   = 1,
            max_mask_length   = 1
        )
    
    @property
    def text_encoder_file(self):
        return os.path.join(self.save_dir, 'text_encoder.json')
    
    @property
    def vocab(self):
        return self.text_encoder.vocab

    @property
    def vocab_size(self):
        return self.text_encoder.vocab_size

    @property
    def blank_token_idx(self):
        return self.text_encoder.blank_token_idx

    @property
    def sep_token(self):
        return self.text_encoder.sep_token

    @property
    def sep_token_idx(self):
        return self.text_encoder.sep_token_idx
    
    @property
    def mask_token_idx(self):
        return self.text_encoder.mask_token_idx
    
    @property
    def sos_token_idx(self):
        return self.text_encoder.sos_token_idx

    @property
    def eos_token_idx(self):
        return self.text_encoder.eos_token_idx

    def __str__(self):
        des = super().__str__()
        des += "- Input language : {}\n".format(self.lang)
        des += "- Input vocab (size = {}) : {}\n".format(self.vocab_size, self.vocab[:50])
        des += "- Input format : {}\n".format(self.input_format)
        des += "- Output format : {}\n".format(self.output_format)
        return des
    
    def encode_text(self, text, * args, ** kwargs):
        return self.text_encoder.encode(text, * args, ** kwargs)
    
    def decode_text(self, encoded, ** kwargs):
        return self.text_encoder.decode(encoded, ** kwargs)
    
    def format_input(self, question = None, context = None, title = None, ** kwargs):
        return self.text_encoder.format(
            self.input_format, question = question, context = context, title = title, ** kwargs
        )
    
    def format_output(self, question = None, context = None, title = None, answer = None,
                      ** kwargs):
        return self.text_encoder.format(
            self.output_format, question = question, context = context, title = title,
            answer = answer, ** kwargs
        )
    
    def tf_encode_text(self, text, default_key = 'text'):
        if isinstance(text, (dict, pd.Series)): text = text[default_key]
        
        encoded_text = tf.py_function(
            self.encode_text, [text], Tout = tf.int32
        )
        encoded_text.set_shape([None])
        
        return encoded_text

    
    def tf_format_input(self, data):
        encoded_text, token_types = tf.py_function(
            self.format_input,
            [data.get('question', ''), data.get('context', ''), data.get('title', '')],
            Tout = [tf.int32, tf.int32]
        )
        encoded_text.set_shape([None])
        token_types.set_shape([None])
        
        return encoded_text, token_types
    
    def tf_format_output(self, data):
        encoded_text, token_types = tf.py_function(
            self.format_output,
            [data.get('question', ''), data.get('context', ''), data.get('title', ''), data.get('answers', '')],
            Tout = [tf.int32, tf.int32]
        )
        encoded_text.set_shape([None])
        token_types.set_shape([None])
        
        return encoded_text, token_types
    
    def get_input(self, data):
        tokens, _ = self.tf_format_input(data)
        
        return (tokens, len(tokens))
    
    def get_output(self, data, inputs = None):
        tokens, _ = self.tf_format_output(data)
        
        return (tokens, len(tokens))
    
    def augment_text(self, tokens, length, min_idx = 1, max_idx = -1, nb_mask = None,
                     min_mask_length = None, max_mask_length = None):
        if nb_mask is None: nb_mask = self.nb_mask
        if min_mask_length is None: min_mask_length = self.min_mask_length
        if max_mask_length is None: max_mask_length = self.max_mask_length
        
        tokens = tf.cond(
            tf.random.uniform(()) < self.augment_prct,
            lambda: random_mask(
                tokens, self.mask_token_idx,
                min_idx = min_idx, max_idx = max_idx,
                nb_mask = nb_mask,
                min_mask_length = min_mask_length,
                max_mask_length = max_mask_length
            ),
            lambda: tokens
        )
        return tokens, len(tokens)

    def encode_data(self, data):
        inputs = self.get_input(data)

        outputs = self.get_output(data, inputs)

        return inputs, outputs
    
    def filter_data(self, inputs, outputs):
        return inputs[1] <= self.max_input_length
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config['lang']      = self.lang
        config['input_format']  = self.input_format
        config['output_format'] = self.output_format
        
        config['text_encoder']  = self.text_encoder_file
        
        config['max_input_length']  = self.max_input_length
        config['use_fixed_length_input']    = self.use_fixed_length_input
        
        return config
        
