
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
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm

from utils import pad_batch, save_embeddings
from utils.text import get_encoder, random_mask
from models.base_model import BaseModel
from models.qa.base_qa import find_index
from custom_architectures.transformers_arch import get_pretrained_transformer_encoder

class ContextRetriever(BaseModel):
    def __init__(self,
                 lang,
                 
                 threshold      = 0.5,
                 distance_metric    = 'euclidian',
                 
                 is_binary_loss = None,
                 pretrained_qe  = 'facebook/dpr-question_encoder-single-nq-base',
                 pretrained_ce  = 'facebook/dpr-ctx_encoder-single-nq-base',
                 
                 add_title      = True,
                 split_on_sentence      = False,
                 max_question_length    = 128,
                 max_context_length     = 512,
                 fixed_length_context   = False,
                 
                 context_text_encoder   = None,
                 question_text_encoder  = None,
                 
                 ** kwargs
                ):
        self.lang   = lang
        
        self.add_title      = add_title
        self.split_on_sentence  = split_on_sentence
        self.max_question_length    = max_question_length
        self.max_context_length     = max_context_length
        self.fixed_length_context   = fixed_length_context
        
        self.threshold      = threshold
        self.distance_metric    = distance_metric
        
        self.is_binary_loss     = is_binary_loss
        
        # Initialization of Text Encoder
        self.question_text_encoder  = get_encoder(text_encoder = question_text_encoder, lang = lang)
        self.context_text_encoder   = get_encoder(text_encoder = context_text_encoder, lang = lang)

        kwargs.setdefault('pretrained_name', '{} + {}'.format(pretrained_qe, pretrained_ce))
        super().__init__(pretrained_qe = pretrained_qe, pretrained_ce = pretrained_ce, ** kwargs)
        
        # Saving text encoder
        if not os.path.exists(self.question_text_encoder_file):
            self.question_text_encoder.save_to_file(self.question_text_encoder_file)
        if not os.path.exists(self.context_text_encoder_file):
            self.context_text_encoder.save_to_file(self.context_text_encoder_file)

    def init_train_config(self,
                          max_question_length   = None,
                          
                          max_context_length    = None,
                          
                          nb_mask   = 1,
                          min_mask_length   = 1,
                          max_mask_length   = 1,
                          
                          max_negatives = -1,
                          in_batch_negatives    = True,
                          
                          ** kwargs
                         ):
        if max_question_length: self.max_question_length = max_question_length
        if max_context_length: self.max_context_length = max_context_length
        
        self.nb_mask = nb_mask
        self.min_mask_length    = min_mask_length
        self.max_mask_length    = max_mask_length
        
        self.max_negatives  = max_negatives
        self.in_batch_negatives = in_batch_negatives
        
        super().init_train_config(** kwargs)

    def build_encoder(self, embedding_dim = 512, pretrained = 'bert-base-uncased', ** kwargs):
        return get_pretrained_transformer_encoder(
            pretrained, output_dim = embedding_dim, return_attention = False, ** kwargs
        )

    def build_question_encoder(self, embedding_dim, pretrained_qe = None, pretrained_ce = None, ** kwargs):
        config = {k[3:] : v for k, v in kwargs.items() if k.startswith('qe_')}
        return self.build_encoder(
            pretrained = pretrained_qe, embedding_dim = embedding_dim, name = 'question_encoder', ** config
        )
    
    def build_context_encoder(self, embedding_dim, pretrained_qe = None, pretrained_ce = None, ** kwargs):
        config = {k[3:] : v for k, v in kwargs.items() if k.startswith('ce_')}
        return self.build_encoder(
            pretrained = pretrained_ce, embedding_dim = embedding_dim, name = 'context_encoder', ** config
        )

    def _build_model(self, normalize = True, ** kwargs):
        def add_normalization(encoder, is_question_encoder):
            if isinstance(encoder, tf.keras.Sequential):
                if normalize:
                    encoder.add(tf.keras.layers.Lambda(
                        l2_normalize, name = 'normalization_layer'
                    ))
                
                signature = encoder.input_shape[1:]
            else:
                if normalize:
                    logging.warning("Encoder is not a `tf.keras.Sequential` so you have to handle `normalize` internally !")
                signature = self.question_input_signature if is_question_encoder else self.context_input_signature
                
            return signature
            
        question_encoder    = kwargs.pop('question_encoder', self.build_question_encoder(** kwargs))
        context_encoder     = kwargs.pop('context_encoder', self.build_context_encoder(** kwargs))
        
        input_kwargs    = {
            'input_signature_a' : add_normalization(question_encoder, True),
            'input_signature_b' : add_normalization(context_encoder, False)
        }
        
        comparator_config = {
            'architecture_name' : 'comparator',
            'encoder_a'     : question_encoder,
            'encoder_b'     : context_encoder,
            'distance_metric'   : self.distance_metric,
            ** input_kwargs,
            ** kwargs
        }
                
        super()._build_model(comparator = comparator_config)

    @property
    def question_text_encoder_file(self):
        return os.path.join(self.save_dir, 'question_text_encoder.json')

    @property
    def context_text_encoder_file(self):
        return os.path.join(self.save_dir, 'context_text_encoder.json')

    @property
    def question_input_signature(self):
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),  # tokens
            tf.TensorSpec(shape = (None, ), dtype = tf.int32)
        )
    
    @property
    def context_input_signature(self):
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),  # tokens
            tf.TensorSpec(shape = (None, ), dtype = tf.int32)
        )

    @property
    def embedding_dim(self):
        return self.question_encoder.output_shape[-1]
    
    @property
    def input_signature(self):
        return (self.question_input_signature, self.context_input_signature)
    
    @property
    def output_signature(self):
        return tf.TensorSpec(shape = (None,), dtype = tf.int32)
    
    @property
    def question_encoder(self):
        return self.comparator.layers[-3]
    
    @property
    def context_encoder(self):
        return self.comparator.layers[-2]
    
    @property
    def decoder(self):
        return self.comparator.layers[-1]
    
    @property
    def training_hparams(self):
        return super().training_hparams(
            max_question_length   = None,
            max_context_length    = None,
            nb_mask   = 1,
            max_negatives   = -1,
            min_mask_length   = 1,
            max_mask_length   = 1,
            in_batch_negatives    = True,
        )
    
    @property
    def question_vocab(self):
        return self.question_text_encoder.vocab

    @property
    def context_vocab(self):
        return self.context_text_encoder.vocab

    @property
    def question_vocab_size(self):
        return self.question_text_encoder.vocab_size

    @property
    def context_vocab_size(self):
        return self.context_text_encoder.vocab_size

    @property
    def question_blank_token_idx(self):
        return self.question_text_encoder.blank_token_idx

    @property
    def context_blank_token_idx(self):
        return self.context_text_encoder.blank_token_idx

    @property
    def question_mask_token_idx(self):
        return self.question_text_encoder.mask_token_idx
    
    @property
    def context_mask_token_idx(self):
        return self.context_text_encoder.mask_token_idx

    @property
    def context_sos_token_idx(self):
        return self.context_text_encoder.sos_token_idx
    
    @property
    def context_eos_token_idx(self):
        return self.context_text_encoder.eos_token_idx
    
    @property
    def context_sep_token_idx(self):
        return self.context_text_encoder[self.context_text_encoder.sep_token]
    
    
    def __str__(self):
        des = super().__str__()
        des += "Input language : {}\n".format(self.lang)
        des += "Embedding dim : {}\n".format(self.embedding_dim)
        des += "Distance metric : {}\n".format(self.distance_metric)
        des += "Question vocab (size = {}) : {}\n".format(self.question_vocab_size, self.question_vocab[:25])
        des += "Context vocab (size = {}) : {}\n".format(self.context_vocab_size, self.context_vocab[:25])
        return des
    
    def call(self, inputs, training = False, ** kwargs):
        question, context = inputs
        
        batch_size  = tf.shape(question[0])[0]
        n   = batch_size if self.in_batch_negatives else tf.shape(context[0])[1]
        
        if not self.in_batch_negatives:
            c_tokens, c_length = context
            c_tokens    = tf.reshape(c_tokens, [batch_size * n, tf.shape(c_tokens)[-1]])
            c_length    = tf.reshape(c_length, [batch_size * n, 1])
            context = (c_tokens, c_length)

        q_emb = self.question_encoder(question, training = training)
        c_emb = self.context_encoder(context, training = training)
            
        if not self.in_batch_negatives:
            c_emb = tf.reshape(c_emb, [batch_size, n, tf.shape(q_emb)[1]])
        
        return self.decoder([q_emb, c_emb], training = training, pred_matrix = True)
    
    def similarity(self, questions, contexts, max_matrix_size = 10000000, ** kwargs):
        return self.decoder([questions, contexts], max_matrix_size = max_matrix_size, ** kwargs)
    
    def compile(self, loss = 'binary_crossentropy', metrics = None, loss_config = {}, ** kwargs):
        self.is_binary_loss = (loss == 'binary_crossentropy')
        if metrics is None:
            metrics = ['binary_accuracy', 'EER'] if self.is_binary_loss else ['sparse_categorical_accuracy']
        
        super().compile(loss = loss, metrics = metrics, loss_config = loss_config, ** kwargs)
    
    def decode_context(self, encoded):
        return np.array(self.context_text_encoder.decode(encoded))
    
    def tf_decode_context(self, encoded_context):
        decoded = tf.py_function(self.decode_context, [encoded_context], Tout = tf.string)
        decoded.set_shape([None])
        return decoded
    
    def get_question(self, text, * args, ** kwargs):
        if isinstance(text, pd.DataFrame): text = text.to_dict('records')
        if isinstance(text, (list, tuple, np.ndarray)):
            tokens  = [self.get_question(row) for row in text]
            lengths = [len(tok) for tok in tokens]
            return pad_batch(tokens, pad_value = self.question_blank_token_idx), np.array(lengths)
        if isinstance(text, (dict, pd.Series)):
            text = text['answers'] if 'answers' in text else text['text']
        return self.question_text_encoder.encode(text, * args, ** kwargs)
    
    def get_context(self, text, title = None, answers = None, ** kwargs):
        if isinstance(text, pd.DataFrame): text = text.to_dict('records')
        if isinstance(text, (list, tuple, np.ndarray)):
            texts, lengths = [], []
            for data in text:
                t, l = self.get_context(data)
                texts.append(t)
                lengths.append(l)
            return pad_batch(texts, pad_value = self.contexts_blank_token_idx), np.concatenate(lengths)
        
        if isinstance(text, (dict, pd.Series)):
            if title is None and 'title' in text: title = text['title']
            if answers is None and 'answers' in text: answers = text['answers']
            text = text['context'] if 'context' in text else text['text']
        
        if isinstance(text, tf.Tensor): text = text.numpy().decode('utf-8')
        if isinstance(answers, tf.Tensor): answers = answers.numpy().decode('utf-8')
        
        prefix = title if self.add_title else None
        if isinstance(prefix, tf.Tensor): prefix = prefix.numpy().decode('utf-8')

        encoded = self.context_text_encoder.split(
            text, max_length = self.max_context_length, prefix = prefix, not_split = answers
        )
        
        lengths = [len(enc) for enc in encoded]
        return pad_batch(encoded, pad_value = self.context_blank_token_idx), lengths
    
    def get_answer(self, text):
        if isinstance(text, tf.Tensor): text = text.numpy().decode('utf-8')
        return self.context_text_encoder.encode(' {} '.format(text.strip()), add_sos_and_eos = False)
    
    def decode_output(self, output):
        return output > self.threshold
    
    def embed_single_question(self, question, ** kwargs):
        tokens = self.encode_question(question)
        return self.question_encoder([tf.expand_dims(tokens, 0), [len(tokens)]], ** kwargs)
    
    def embed_single_context(self, context, title, ** kwargs):
        tokens, lengths = self.encode_context({'context' : context, 'title' : title})
        return self.context_encoder([tokens, lengths], ** kwargs)
    
    def embed_question(self, questions, batch_size = 128, tqdm = lambda x: x, ** kwargs):
        if isinstance(questions, pd.DataFrame):
            key = 'question' if 'question' in questions.columns else 'text'
            questions = questions[key].values
        if isinstance(questions, tf.Tensor): questions = questions.numpy().decode('utf-8')
        if not isinstance(questions, (list, tuple, np.ndarray)): questions = [questions]

        embeddings = []
        tokens = [self.get_question(q) for q in questions]
        for start in tqdm(range(0, len(tokens), batch_size)):
            batch_tokens = tokens[start : start + batch_size]
            lengths      = np.array([len(t) for t in batch_tokens])
            batch_tokens = pad_batch(batch_tokens, pad_value = self.question_blank_token_idx)

            embedded = self.question_encoder([batch_tokens, lengths], training = False)
            embeddings.append(embedded)

        embeddings = np.concatenate(embeddings, axis = 0)

        return pd.DataFrame([{'question' : q, 'question_embedding' : e} for q, e in zip(questions, embeddings)])    

    def embed_context(self, context, title = None, ids = None, answer = None, split_context = True,
                      batch_size = 128, tqdm = lambda x: x, ** kwargs):
        def _to_list(x):
            if x is None: return x
            if isinstance(x, tf.Tensor): x = x.numpy().decode('utf-8')
            if not isinstance(x, (list, tuple, np.ndarray)): x = [x]
            return x

        splitted_ids, splitted_titles, splitted_contexts, tokens = [], [], [], []

        if not split_context:
            if isinstance(context, pd.DataFrame):
                if 'title' in context.columns: title = context['title'].values
                if 'context_id' in context.columns: ids = context['context_id']
                elif 'id' in context.columns: ids = context['id'].values
                context = context['context'].values if 'context' in context else context['text'].values

            context = _to_list(context)
            title   = _to_list(title)
            ids     = _to_list(ids)

            for i, ci in enumerate(context):
                ti =title[i] if title is not None and self.add_title else None

                if ti:
                    encoded = self.context_text_encoder.join(ti, ci)[0]
                else:
                    encoded = self.context_text_encoder.encode(context)
                tokens.append(encoded)

            splitted_ids, splitted_titles, splitted_contexts = ids, title, context

        else:
            if isinstance(context, pd.DataFrame):
                id_key = 'context_id' if 'context_id' in context.columns else 'id' if 'id' in context.columns else None
                c_key = 'context' if 'context' in context.columns else 'text'

                if id_key is not None:
                    group_keys = id_key
                elif self.add_title and 'title' in context.columns:
                    group_keys = ['title', c_key]
                else:
                    group_keys = [c_key]

                unique_contexts, title, answer, ids = [], [], [], []
                for k, data in context.groupby(group_keys):
                    item = data.iloc[0]
                    if id_key is not None: ids.append(item[id_key])
                    if 'title' in data.columns and self.add_title: title.append(item['title'])
                    if 'answers' in data.columns: answer.append(data['answers'].unique())
                    unique_contexts.append(item[c_key])

            else:
                unique_contexts = _to_list(context)
                title = _to_list(title)
                answer = _to_list(answer)
                ids = _to_list(ids)

            if not ids: ids = np.arange(len(unique_contexts))

            for i, ci in enumerate(tqdm(unique_contexts)):
                prefix =title[i] if title is not None and self.add_title else None
                not_split = answer[i] if answer is not None else None

                encoded = self.context_text_encoder.split(
                    ci, max_length = self.max_context_length, prefix = prefix, not_split = not_split
                )

                start_ctx_idx = 1 if prefix is None else (np.where(encoded[0] == self.context_sep_token_idx)[0][0] + 1)

                splitted_ids.extend([ids[i]] * len(encoded))
                splitted_titles.extend([prefix] * len(encoded))
                splitted_contexts.extend([
                    self.context_text_encoder.decode(enc[start_ctx_idx : -1]) for enc in encoded
                ])
                tokens.extend(encoded)

        embeddings = []
        for start in tqdm(range(0, len(tokens), batch_size)):
            batch   = tokens[start : start + batch_size]
            lengths = tf.cast([len(t) for t in batch], tf.int32)
            batch   = pad_batch(batch, pad_value = self.context_blank_token_idx)

            embedded = self.context_encoder([batch, lengths], training = False).numpy()
            embeddings.append(embedded)

        embeddings = np.concatenate(embeddings, axis = 0)

        return pd.DataFrame([
            {'context_id' : c_id, 'title' : t, 'context' : c, 'context_embedding' : e}
            for c_id, t, c, e in zip(splitted_ids, splitted_titles, splitted_contexts, embeddings)
        ])
    
    def encode_question(self, data):
        text = data['question'] if isinstance(data, (dict, pd.Series)) else data
        tokens = tf.py_function(self.get_question, [text], Tout = tf.int32)
        tokens.set_shape([None])
        return tokens
    
    def encode_context(self, data):
        text    = data['context'] if isinstance(data, (dict, pd.Series)) else data
        title   = data['title'] if isinstance(data, (dict, pd.Series)) else ''
        answer  = data['answers'] if isinstance(data, (dict, pd.Series)) else ''
        
        tokens, lengths = tf.py_function(self.get_context, [text, title, answer], Tout = [tf.int32, tf.int32])
        tokens.set_shape([None, None])
        lengths.set_shape([None])
        
        return tokens, lengths
    
    def encode_answer(self, data):
        text = data['answers'] if isinstance(data, (dict, pd.Series)) else data
        tokens = tf.py_function(self.get_answer, [text], Tout = tf.int32)
        tokens.set_shape([None])
        return tokens
    
    def augment_question(self, inp):
        return inp
    
    def augment_context(self, inp):
        if not self.in_batch_negatives or True: return inp
        tokens, length = inp
        tokens = tf.cond(
            tf.random.uniform(()) < self.augment_prct,
            lambda: random_mask(
                tokens, self.context_mask_token_idx, min_idx = 1, max_idx = len(inp) - 1, nb_mask = self.nb_mask,
                min_mask_length = self.min_mask_length, max_mask_length = self.max_mask_length
            ),
            lambda: tokens
        )
        return tokens, len(tokens)
    
    def preprocess_question(self, question, context):
        return question
    
    def preprocess_context(self, question, context):
        return context
    
    def encode_data(self, data):
        question_tokens = self.encode_question(data)
        context_tokens, context_lengths = self.encode_context(data)
        
        answer_tokens   = self.encode_answer(data)
        index = find_index(tf.reshape(context_tokens, [-1]), answer_tokens)
        
        valid_idx   = (index // tf.shape(context_tokens)[1]) if index != -1 else -1
        
        n_contexts  = tf.shape(context_lengths)[0]
        
        if self.max_negatives >= 0 and n_contexts - 1 > self.max_negatives:
            indexes = tf.random.shuffle(tf.range(n_contexts))[:self.max_negatives]
            if valid_idx != -1 and not tf.reduce_any(indexes == valid_idx):
                indexes = tf.concat([indexes, [valid_idx]], axis = 0)

            context_tokens  = tf.gather(context_tokens, indexes)
            context_lengths = tf.gather(context_lengths, indexes)
            if valid_idx != -1: valid_idx = tf.cast(tf.where(indexes == valid_idx)[0,0], tf.int32)
                
        if self.max_negatives >= 0 and valid_idx != -1:
            str_context = self.tf_decode_context(context_tokens)
        else:
            str_context = tf.expand_dims(data['context'], axis = 0)
        
        return ((question_tokens, len(question_tokens)), (context_tokens, context_lengths, str_context)), valid_idx
    
    def filter_data(self, inputs, target):
        question, context = inputs
        return target != -1 and question[1] <= self.max_question_length # and tf.reduce_max(context[1]) <= self.max_context_length
    
    def augment_data(self, inputs, target):
        question, context = inputs
        question    = self.augment_question(question)
        context     = self.augment_context(context)
        
        return (question, context), target
        
    def preprocess_data(self, inputs, target):
        (q_tokens, q_lengths), (c_tokens, c_lengths, str_context) = inputs

        if self.in_batch_negatives:
            str_context = tf.reshape(str_context, [-1])

            if self.max_negatives < 0:
                _, idx = tf.unique(str_context)
                c_idx, _ = tf.unique(idx)

                # `idx` is for instance [0, 1, 1, 0, 2]
                # so we should keep indexes [0, 1, 4] (because indexes [2 and 3] are duplicates)
                # `positions` is, for each unique index [0, 1, 2] an array of positions [0, 1, 2, 3, 4]
                positions = tf.range(0, tf.shape(idx)[0])
                positions = tf.tile(tf.expand_dims(positions, 1), [1, tf.shape(c_idx)[0]])
                mask = tf.cast(c_idx == tf.expand_dims(idx, 1), tf.int32)

                # then we get the 1st position where each unique id appears [0, 1, 4]
                real_idx = tf.reduce_min(positions * mask + (tf.shape(mask)[0] * (1 - mask)), axis = 0)

                # Adjust target indexes
                # Suppose we have 3 unique contexts with n_parts = [1, 2, 3]
                # the target = 0 for the 2nd paragraph becomes index 1 (because index 0 is the 1st paragraph of 1st context-part)
                # then target = 1 becomes the 1st paragraph of the 2nd context-part (the real target)
                n_valid     = tf.gather(tf.reduce_sum(tf.cast(c_lengths > 0, tf.int32), axis = -1), real_idx)
                n_valid     = tf.gather(tf.cumsum(n_valid) - n_valid, idx)
                target      = target + n_valid

                # We extract unique context
                c_tokens    = tf.gather(c_tokens, real_idx)
                c_lengths   = tf.gather(c_lengths, real_idx)
            else:
                n_valid     = tf.reduce_sum(tf.cast(c_lengths > 0, tf.int32), axis = -1)
                target      = target + tf.cumsum(n_valid) - n_valid


            c_tokens    = tf.reshape(c_tokens, [-1, tf.shape(c_tokens)[-1]])
            c_lengths   = tf.reshape(c_lengths, [-1])
            # Then we remove context with 0-length (batch padding)
            valid_idx   = c_lengths > 0
            c_tokens    = tf.boolean_mask(c_tokens, valid_idx)
            c_lengths   = tf.boolean_mask(c_lengths, valid_idx)

            if self.max_negatives >= 0:
                str_context = tf.boolean_mask(str_context, valid_idx)

                _, idx = tf.unique(str_context)
                c_idx, _ = tf.unique(idx)

                # `idx` is for instance [0, 1, 1, 0, 2]
                # so we should keep indexes [0, 1, 4] (because indexes [2 and 3] are duplicates)
                # `positions` is, for each unique index [0, 1, 2] an array of positions [0, 1, 2, 3, 4]
                positions = tf.range(0, tf.shape(idx)[0])
                positions = tf.tile(tf.expand_dims(positions, 1), [1, tf.shape(c_idx)[0]])
                mask = tf.cast(c_idx == tf.expand_dims(idx, 1), tf.int32)

                # then we get the 1st position where each unique id appears [0, 1, 4]
                real_idx = tf.reduce_min(positions * mask + (tf.shape(mask)[0] * (1 - mask)), axis = 0)

                # Adjust target indexes
                # Suppose we have 3 unique contexts with n_parts = [1, 2, 3]
                # the target = 0 for the 2nd paragraph becomes index 1 (because index 0 is the 1st paragraph of 1st context-part)
                # then target = 1 becomes the 1st paragraph of the 2nd context-part (the real target)
                #target     = tf.gather(tf.range(tf.shape(c_context)[0])), target)
                target      = tf.gather(idx, target) #target + n_valid

                # We extract unique context
                c_tokens    = tf.gather(c_tokens, real_idx)
                c_lengths   = tf.gather(c_lengths, real_idx)

        return ((q_tokens, q_lengths), (c_tokens, c_lengths)), target
        
    def get_dataset_config(self, ** kwargs):
        kwargs['batch_before_map']  = True
        
        kwargs['padded_batch']  = True
        kwargs['pad_kwargs']    = {
            'padding_values' : (
                ((self.question_blank_token_idx, 0), (self.context_blank_token_idx, 0, '')), 0
            )
        }
        return super().get_dataset_config(** kwargs)
    
    def train_step(self, batch):
        inputs, target = batch

        with tf.GradientTape() as tape:
            dist_matrix = self(inputs, training = True)
            
            if self.is_binary_loss:
                # Reshape to shape [batch_size * n, 1] for binary comparison
                target      = tf.reshape(tf.one_hot(target, tf.shape(dist_matrix)[1]), [-1, 1])
                dist_matrix = tf.reshape(dist_matrix, [-1, 1])

            loss = self.comparator_loss(target, dist_matrix)
        
        grads = tape.gradient(loss, self.comparator.trainable_variables)
        self.comparator_optimizer.apply_gradients(zip(grads, self.comparator.trainable_variables))

        return self.update_metrics(target, dist_matrix)
    
    def eval_step(self, batch):
        inputs, target = batch
        
        dist_matrix = self(inputs, training = False)
        
        if self.is_binary_loss:
            # Reshape to shape [batch_size * n, 1] for binary comparison
            target      = tf.reshape(tf.one_hot(target, tf.shape(dist_matrix)[1]), [-1, 1])
            dist_matrix = tf.reshape(dist_matrix, [-1, 1])

        return self.update_metrics(target, dist_matrix)
    
    def retrieve(self, questions, contexts, k = 10, tqdm = lambda x: x, ** kwargs):
        embedded_questions = self.embed_question(questions, tqdm = tqdm, ** kwargs) if 'question_embedding' not in questions else questions
        embedded_contexts  = self.embed_context(contexts, tqdm = tqdm, ** kwargs) if 'context_embedding' not in contexts else contexts

        q_embeddings = np.array([e for e in embedded_questions['question_embedding'].values])
        c_embeddings = np.array([e for e in embedded_contexts['context_embedding'].values])

        similarities    = self.similarity(q_embeddings, c_embeddings, pred_matrix = True)
        indexes         = tf.argsort(similarities, axis = -1, direction = 'DESCENDING').numpy()
        similarities    = similarities.numpy()

        result = []
        for i in range(len(indexes)):
            result_i = []
            for ki in range(k):
                idx = indexes[i, ki]
                result_i.append({
                    'index'      : idx,
                    'context_id' : embedded_contexts.at[idx, 'context_id'],
                    'title'      : embedded_contexts.at[idx, 'title'],
                    'context'    : embedded_contexts.at[idx, 'context'],
                    'score'      : similarities[i, idx]
                })
            result.append(result_i)

        return q_embeddings, c_embeddings, result
    
    def embed_dataset(self, questions, contexts = None, tqdm = lambda x: x, path = None, ** kwargs):
        if 'context_id' not in questions:
            assert 'context' in questions.columns
            questions = pd.merge(
                pd.DataFrame([
                    {'context_id' : i, 'context' : c}
                    for i, c in enumerate(questions['context'].unique())
                ]),
                questions,
                on = 'context'
            )
        if contexts is None: contexts = questions
        if 'context_id' not in contexts:
            contexts['context_id'] = contexts['id'].values if 'id' in contexts.columns else np.arange(len(contexts))

        embedded_questions = self.embed_question(questions, tqdm = tqdm, ** kwargs)
        embedded_contexts  = self.embed_context(contexts, tqdm = tqdm, ** kwargs)

        embedded_questions  = pd.merge(
            embedded_questions, questions[['question', 'answers', 'context_id']], on = 'question'
        )

        embeddings = pd.merge(embedded_questions, embedded_contexts, on = 'context_id')
        embeddings['valid_context'] = embeddings.apply(
            lambda row: self.context_text_encoder.invert(row['answers']) in row['context'], axis = 'columns'
        )
        
        if path is not None:
            save_embeddings(path, embeddings, embedding_name = '{}_embeddings.csv'.format(self.nom))

        return embeddings
    
    def evaluate(self, questions, contexts = None, max_k = 100, tqdm = lambda x: x, ** kwargs):
        if 'context_id' not in questions:
            assert 'context' in questions.columns
            questions = pd.merge(
                pd.DataFrame([{'context_id' : i, 'context' : c} for i, c in enumerate(questions['context'].unique())]),
                questions,
                on = 'context'
            )
        if contexts is None: contexts = questions
        if 'context_id' not in contexts:
            contexts['context_id'] = contexts['id'].values if 'id' in contexts.columns else np.arange(len(contexts))

        _, _, results = self.retrieve(questions, contexts, k = max_k, tqdm = tqdm, ** kwargs)

        answers = questions[['context_id', 'answers']].to_dict('records')

        found = np.zeros((max_k,))
        for i, result_i in enumerate(tqdm(results)):
            if not isinstance(answers[i]['context_id'], (list, tuple)):
                answers[i]['context_id'] = [answers[i]['context_id']]
            if kwargs.get('split_context', True):
                answers[i]['answers'] = self.context_text_encoder.invert(answers[i]['answers'])
            
            for k, context in enumerate(result_i):
                if context['context_id'] in answers[i]['context_id'] and answers[i]['answers'] in context['context']:
                    found[k] += 1
                    break

        prct = np.cumsum(found) / len(results)
        return prct

    
    def get_config(self, *args, ** kwargs):
        """ Return base configuration for a `siamese network` """
        config = super().get_config(*args, **kwargs)
        config['lang']   = self.lang
        
        config['add_title']     = self.add_title
        config['split_on_sentence'] = self.split_on_sentence
        config['max_question_length']   = self.max_question_length
        config['max_context_length']    = self.max_context_length
        config['fixed_length_context']  = self.fixed_length_context
        
        config['threshold']     = self.threshold
        config['distance_metric']   = self.distance_metric
        
        config['is_binary_loss']    = self.is_binary_loss
        
        config['question_text_encoder'] = self.question_text_encoder_file
        config['context_text_encoder']  = self.context_text_encoder_file
        
        return config