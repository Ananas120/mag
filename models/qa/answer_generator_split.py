import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer
from utils import pad_batch
from models.qa.base_qa_generator import BaseQAGenerator

class AnswerGeneratorSplit(BaseQAGenerator):
    def __init__(self,
                 * args,
                 
                 input_format   = None,
                 question_format    = '{question}',
                 context_format     = '{context}',
                 context_offset     = -1,
                 subsample_question = True,
                 skip_question_eos  = False,
                 skip_context_sos   = False,
                 
                 ** kwargs
                ):
        self.question_format    = question_format
        self.context_format     = context_format
        self.context_offset     = context_offset
        
        self.skip_question_eos  = skip_question_eos
        self.skip_context_sos   = skip_context_sos
        
        self.subsample_question = subsample_question
        
        self.force_merging      = False
        
        self.__memory   = None
        
        super().__init__(* args, input_format = None, ** kwargs)
    
    def init_train_config(self, negative_mode = 'batch', max_negatives = -1, augment_question = False, ** kwargs):
        assert negative_mode in (None, 'none', 'batch', 'doc')
        if negative_mode == 'none': negative_mode = None
        
        self.negative_mode      = negative_mode
        self.max_negatives      = max_negatives if self.use_document else -1
        self.augment_question   = augment_question
        
        super().init_train_config(** kwargs)
    
    @property
    def training_hparams(self):
        return super().training_hparams(negative_mode = 'batch', max_negatives = -1, augment_question = False)
    
    @property
    def context_shape(self):
        return ((None, None), (None, )) if not self.use_document else ((None, None, None), (None, None))
        
    @property
    def input_signature(self):
        ctx_shape, ctx_len_shape = self.context_shape
        
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),  # text (tokens ids) for encoder input
            tf.TensorSpec(shape = (None,), dtype = tf.int32),       # text length for encoder input
            tf.TensorSpec(shape = ctx_shape, dtype = tf.int32),  # text (tokens ids) for encoder input
            tf.TensorSpec(shape = ctx_len_shape, dtype = tf.int32),       # text length for encoder input
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),  # text (tokens ids) for decoder input
            tf.TensorSpec(shape = (None,), dtype = tf.int32)        # text length for decoder input
        )
    
    @property
    def encoder(self):
        return self.model.encoder
    
    @property
    def decoder(self):
        return self.model.decoder
    
    @property
    def in_batch_negatives(self):
        return self.negative_mode == 'batch'
    
    @property
    def use_document(self):
        return self.negative_mode == 'doc'
    
    def __str__(self):
        des = super().__str__()
        des += "- Question format : {}\n".format(self.question_format)
        des += "- Context format : {}\n".format(self.context_format)
        return des

    def __call__(self, * args, ** kwargs):
        if self.use_document:
            return self.call(* args, ** kwargs)
        return self.call_fn(* args, ** kwargs)
    
    @timer(log_if_root = False)
    def encode(self, inputs, training = False, merge_contexts = False, verbose = False):
        assert len(inputs) % 2 == 0 and len(inputs) >= 2
        
        q_encoded, q_mask = self.encoder(
            [inputs[0], inputs[1]], training = training, force_not_subsampling = not self.subsample_question,
            return_attention = False, return_mask = True
        )
        
        if len(inputs) == 2: return (encoded, mask)

        
        encodings, masks = [], []
        for i in range(2, len(inputs), 2):
            tokens, length = inputs[i], inputs[i + 1]
            
            if verbose:
                tf.print("Context", i, "shape :", tf.shape(tokens))
            
            if len(tf.shape(tokens)) == 3:
                tokens  = tf.reshape(tokens, [-1, tf.shape(tokens)[-1]])
                lengths = tf.reshape(lengths, [-1])
            
            encoded, mask = self.encoder(
                [tokens, length], training = training, positional_offset = self.context_offset,
                return_attention = False, return_states = False, return_mask = True
            )
            
            if len(tf.shape(tokens)) == 3:
                encoded = tf.reshape(encoded, [tf.shape(q_encoded)[0], -1, tf.shape(encoded)[-1]])
                mask    = tf.reshape(mask, [tf.shape(q_encoded)[0], 1, 1, -1])
            
            if verbose:
                tf.print("Context", i, "encoded shape :", tf.shape(tokens))
            
            encodings.append(encoded)
            masks.append(mask)
        
        contexts    = tf.concat(encodings, axis = 1) if len(inputs) > 4 else encodings[0]
        c_masks     = tf.concat(masks, axis = -1) if len(inputs) > 4 else masks[0]
        
        if verbose:
            tf.print("Encodings shape : {}".format(tf.shape(contexts)))
            tf.print("Masks shape : {}".format(tf.shape(c_masks)))
        
        if merge_contexts or self.force_merging or (self.in_batch_negatives and training):
            contexts = tf.reshape(
                tf.tile(contexts, [tf.shape(contexts)[0], 1, 1]), 
                [tf.shape(contexts)[0], -1, tf.shape(contexts)[-1]]
            )
            c_masks = tf.reshape(
                tf.tile(c_masks, [tf.shape(c_masks)[0], 1, 1, 1]), 
                [tf.shape(c_masks)[0], 1, 1, -1]
            )
            
            if verbose:
                tf.print("Encodings shape : {}".format(tf.shape(contexts)))
                tf.print("Masks shape : {}".format(tf.shape(c_masks)))
        
        encodings   = tf.concat([q_encoded, contexts], axis = 1)
        masks       = tf.concat([q_mask, c_masks], axis = -1)
        
        return (encodings, masks)
    
    @timer(name = 'prediction', log_if_root = False)
    def call(self, inputs, training = False):
        encoder_outputs, enc_padding_mask = self.encode(inputs[:-2], training = training)

        answer_tokens, answer_lengths = inputs[-2 :]
        
        decoder_inputs = (encoder_outputs, answer_tokens, answer_lengths)
        
        return self.decoder(
            decoder_inputs, enc_padding_mask = enc_padding_mask, training = training, return_attention = False
        )
    
    @timer(name = 'inference', log_if_root = False)
    def infer(self, inputs, training = False, ** kwargs):
        kwargs.setdefault('max_length', self.max_output_length)
        encoder_outputs, enc_padding_mask = self.encode(inputs, training = training)

        return self.decoder.infer(
            encoder_outputs, enc_padding_mask = enc_padding_mask, training = training, ** kwargs
        )
    
    def format_question(self, question, ** kwargs):
        formatted = self.text_encoder.format(self.question_format, question = question, ** kwargs)
        if self.skip_question_eos: formatted = formatted[:-1]
        return formatted
    
    def format_context(self, context, title = None, ** kwargs):
        formatted = self.text_encoder.format(self.context_format, context = context, title = title, ** kwargs)
        if self.skip_context_sos: formatted = formatted[1:]
        return formatted

    def encode_document(self, context, title = None, ** kwargs):
        if isinstance(context, tf.Tensor): context = context.numpy()
        if isinstance(title, tf.Tensor): title = title.numpy()
        
        if not isinstance(context, (list, tuple, np.ndarray)): context = [context]
        if title is not None and not isinstance(title, (list, tuple, np.ndarray)): title = [title]
        if title is None: title = [''] * len(context)
        
        paragraphs = [
            self.format_context(c, t)[0] for t, c in zip(title, context)
        ]
        
        return pad_batch(paragraphs, pad_value = self.blank_token_idx), [len(p) for p in paragraphs]
    
    def tf_format_question(self, data):
        q_text = data if not isinstance(data, (dict, pd.Series)) else data.get('question', '')
        
        encoded_text, token_types = tf.py_function(
            self.format_question, [q_text], Tout = [tf.int32, tf.int32]
        )
        encoded_text.set_shape([None])
        
        return encoded_text

    def tf_format_context(self, data):
        if not isinstance(data, (dict, pd.Series)): data = {'context' : data}
        
        encoded_text, token_types = tf.py_function(
            self.format_context, [data.get('context', ''), data.get('title', '')], Tout = [tf.int32, tf.int32]
        )
        encoded_text.set_shape([None])
        
        return encoded_text

    def tf_encode_document(self, data):
        para    = data.get('paragraphs', data.get('context', data))
        titles  = data.get('titles', data.get('title', ''))
        
        encoded_doc, lengths    = tf.py_function(
            self.encode_document, [para, titles], Tout = [tf.int32, tf.int32]
        )
        encoded_doc.set_shape([None, None])
        lengths.set_shape([None])
        
        valid_idx = data.get('valid_idx', -1)
        
        valid_ctx   = tf.logical_or(lengths <= self.max_input_length, tf.range(tf.shape(lengths)[0]) == valid_idx)
        
        n_contexts = tf.shape(lengths)[0]
        if self.max_negatives >= 0 and n_contexts - 1 > self.max_negatives:
            indexes = tf.boolean_mask(tf.range(n_contexts), valid_ctx)
            indexes = tf.random.shuffle(indexes)[:self.max_negatives]
            if valid_idx != -1 and not tf.reduce_any(indexes == valid_idx):
                indexes = tf.concat([indexes, [valid_idx]], axis = 0)

            lengths     = tf.gather(lengths, indexes)
            encoded_doc = tf.gather(encoded_doc, indexes)
        else:
            encoded_doc = tf.boolean_mask(encoded_doc, valid_ctx)
            lengths     = tf.boolean_mask(lengths, valid_ctx)
        
        encoded_doc = encoded_doc[:, : tf.reduce_max(lengths)]
        return encoded_doc, lengths
    
    def get_input(self, data):
        q_tokens = self.tf_format_question(data)
        
        if self.use_document:
            contexts, c_lengths = self.tf_encode_document(data)
            
            return (q_tokens, len(q_tokens), contexts, c_lengths)
        if isinstance(data['context'], list):
            contexts = [self.tf_format_context(c) for c in data['context']]
            
            outputs = (q_tokens, len(q_tokens))
            for c in contexts: outputs += (c, len(c))
            
            return outputs
        
        c_tokens = self.tf_format_context(data)
        
        return (q_tokens, len(q_tokens), c_tokens, len(c_tokens))
    
    def filter_data(self, inputs, outputs):
        max_ctx_length = inputs[3] if not self.use_document else tf.reduce_max(inputs[3])
        if tf.shape(inputs[2])[-1] != max_ctx_length:
            tf.print("ctx shape :", tf.shape(inputs[2]), "-", inputs[3])
        return inputs[1] <= self.max_input_length and tf.shape(inputs[2])[-1] <= self.max_input_length and outputs[1] <= self.max_output_length
    
    def augment_data(self, inputs, outputs):
        q_tokens, q_length, c_tokens, c_length = inputs
        
        if self.augment_question:
            q_tokens, q_length = self.augment_text(q_tokens, q_length, nb_mask = 1, max_mask_length = 2)
        if not self.use_document:
            c_tokens, c_length = self.augment_text(c_tokens, c_length)
        
        return (q_tokens, q_length, c_tokens, c_length), outputs

    def get_dataset_config(self, ** kwargs):
        ctx_shape, ctx_len_shape = self.context_shape
        kwargs.update({
            'batch_before_map'  : True,
            'padded_batch'      : True,
            'pad_kwargs'        : {
                'padded_shapes'     : (
                    ((None,), (), ctx_shape[1:], ctx_len_shape[1:]), ((None, ), ())
                ),
                'padding_values'    : (
                    (self.blank_token_idx, 0, self.blank_token_idx, 0), (self.blank_token_idx, 0)
                )
            }
        })
        
        return super(BaseQAGenerator, self).get_dataset_config(** kwargs)

    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        
        config['question_format']   = self.question_format
        config['context_format']    = self.context_format
        config['context_offset']    = self.context_offset
        
        config['subsample_question']    = self.subsample_question
        
        return config