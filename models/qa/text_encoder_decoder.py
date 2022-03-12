
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

import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer
from utils import pad_batch
from models.qa.base_generator import BaseGenerator
from custom_architectures.transformers_arch.bart_arch import Bart

time_logger = logging.getLogger('timer')

class TextEncoderDecoder(BaseGenerator):
    def __init__(self, lang, input_format = None, output_format = None, ** kwargs):
        super().__init__(lang, input_format = None, output_format = None, ** kwargs)

    def _build_model(self, * args, ** kwargs):
        super()._build_model(* args, ** kwargs)

    @property
    def encoder(self):
        return self.model.encoder
    
    @property
    def decoder(self):
        return self.model.decoder
    
    @timer(log_if_root = False)
    def encode(self, text, text_length = None, ** kwargs):
        if isinstance(text, (list, tuple)): text, text_length = text
        if len(tf.shape(text)) == 1: text = tf.expand_dims(text, axis = 0)
        
        if text_length is None: text_length = tf.fill([tf.shape(text)[0]], tf.shape(text)[1])
        elif len(tf.shape(text_length)) == 0: text_length = tf.expand_dims(text_length, 0)
        
        return self.encoder([text, text_length], ** kwargs)
    
    @timer(log_if_root = False)
    def decode(self, embeddings, ** kwargs):
        if len(tf.shape(embeddings)) == 1: embeddings = tf.reshape(embeddings, [1, 1, tf.shape(embeddings)[0]])
        elif len(tf.shape(embeddings)) == 2: embeddings = tf.expand_dims(embeddings, axis = 1)
        
        return self.decoder.infer(embeddings, ** kwargs)

    def get_input(self, data):
        tokens = self.tf_encode_text(data)
        
        return tokens, len(tokens)
    
    def get_output(self, data, inputs = None):
        if inputs is None: inputs = self.get_input(data)
        
        return inputs

    @timer
    def fast_embed(self, data, ** kwargs):
        tokens, length = self.get_input(data)
        return self.encode([tokens, length], ** kwargs)[0]
    
    @timer
    def embed(self, data, batch_size = 64, tqdm = lambda x: x, ** kwargs):
        time_logger.start_timer('processing')
        
        if not isinstance(data, (list, tuple, pd.DataFrame)): data = [data]

        tokens, lengths = [], []
        if isinstance(data, pd.DataFrame):
            for _, row in data.iterrows():
                tok, l = self.get_input(row)
                tokens.append(tok)
                lengths.append(l)
        else:
            for row in data:
                tok, l = self.get_input(row)
                tokens.append(tok)
                lengths.append(l)

        time_logger.stop_timer('processing')
        
        embeddings = []
        for start in tqdm(range(0, len(tokens), batch_size)):
            batch_tokens = tokens[start : start + batch_size]
            batch_lengths   = np.array(lengths[start : start + batch_size])
            batch_tokens = pad_batch(batch_tokens, pad_value = self.blank_token_idx)

            embedded, mask = self.encode([batch_tokens, batch_lengths], training = False, return_mask = True)
            
            n_valid = tf.reduce_sum(1 - tf.cast(tf.squeeze(mask, [1, 2]), tf.int32), axis = -1)
            embeddings.extend([
                embedding_i[: n] for embedding_i, n in zip(embedded.numpy(), n_valid)
            ])

        return embeddings

    
