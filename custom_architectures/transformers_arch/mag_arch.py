
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

import numpy as np
import tensorflow as tf

from loggers import timer
from utils.sequence_utils import pad_to_multiple
from custom_layers import FasterEmbedding
from custom_architectures.transformers_arch.transformer_arch import build_mask, format_output
from custom_architectures.transformers_arch.bart_arch import Bart, BartEncoder, HParamsBart
from custom_architectures.transformers_arch.text_transformer_arch import *

_supported_subsamplings = ('select', 'mean', 'max', 'min', 'dense', 'conv', 'separable')

HParamsMAGEncoder = HParamsTextTransformerEncoder(
    subsample_at    = -1,
    subsample_after = True,
    subsampling_step    = -1,
    subsampling_offset  = 1,
    subsampling_mode    = 'select',
    subsampling_drop_rate   = 0.,

    repeat_pos_idx      = False,
    
    use_type_embedding      = False,
    random_training_type    = True,
    max_types   = 16
)

HParamsMAG  = HParamsBart(
    ** HParamsMAGEncoder.get_config(add_prefix = 'encoder')
)

@timer
def concat_qc(embeddings,
              mask      = None,
              merge_contexts    = False,
              debug     = False,
              ** kwargs
             ):
    question, contexts = embeddings[0], embeddings[1:]
    q_mask, c_masks     = (mask[0], mask[1:]) if mask is not None else (None, None)
    
    c_lengths   = [tf.shape(c)[-2] for c in contexts]
    contexts    = tf.concat(contexts, axis = 1) if len(contexts) > 1 else contexts[0]
    if c_masks is not None:
        if tf.shape(c_masks[0])[-2] > 1:
            c_masks = tuple([tf.reduce_min(m, axis = -2, keepdims = True) for m in c_masks])
        c_masks = tf.concat(c_masks, axis = -1) if len(c_masks) > 1 else c_masks[0]
    if q_mask is not None and tf.shape(q_mask)[-2] > 1:
        q_mask = tf.reduce_min(q_mask, axis = -2, keepdims = True)
    
    lengths     = [tf.shape(question)[1]] + c_lengths
    
    if debug:
        tf.print("Sequence lengths :", lengths)
        tf.print("Question shape :", tf.shape(question))
        tf.print("Contexts shape :", tf.shape(contexts))
        if c_masks is not None:
            tf.print("Masks shape :", tf.shape(c_masks))
    
    n_doc_per_batch = 1
    q_batch_size, c_batch_size = tf.shape(question)[0], tf.shape(contexts)[0]
    
    # flatten contexts from [B, n_doc, ctx_len, emb_dim] to [B, n_doc * ctx_len, emb_dim]
    if len(tf.shape(contexts)) == 4:
        if len(c_lengths) > 1:
            raise NotImplementedError("When passing multiple document / batch at once, you cannot pass multiple contexts, please flatten everything !")

        n_doc_per_batch = tf.shape(contexts)[1]
        
        ctx_types = tf.repeat(tf.range(1, n_doc_per_batch + 1), tf.shape(contexts)[2])
        
        contexts    = tf.reshape(contexts, [c_batch_size, -1, tf.shape(contexts)[-1]])
        if c_masks is not None:
            c_masks = tf.reshape(c_masks, [c_batch_size, 1, 1, -1])

        if debug:
            tf.print("Contexts (after flattening) shape :", tf.shape(contexts))
            if c_masks is not None:
                tf.print("Masks (after flattening) shape :", tf.shape(c_masks))
    elif len(c_lengths) > 1:
        ctx_types   = tf.concat([
            tf.fill([length], i + 1) for i, length in enumerate(c_lengths)
        ], axis = -1)
    else:
        ctx_types   = tf.fill((tf.shape(contexts)[1], ), 1)
    
    # Merge contexts (if required)
    if merge_contexts and q_batch_size > 1 and q_batch_size == c_batch_size:
        if len(c_lengths) > 1:
            raise NotImplementedError("When merging contexts, you can only pass 1 context / batch !")
        
        ctx_add_type = tf.repeat(tf.range(q_batch_size), tf.shape(contexts)[1])

        contexts = tf.reshape(
            tf.tile(contexts, [c_batch_size, 1, 1]), 
            [c_batch_size, -1, tf.shape(contexts)[-1]]
        )
        if c_masks is not None:
            c_masks = tf.reshape(
                tf.tile(c_masks, [c_batch_size, 1, 1, 1]), 
                [c_batch_size, 1, 1, -1]
            )
        
        if debug:
            tf.print("Contexts (after merging) shape :", tf.shape(contexts))
            if c_masks is not None:
                tf.print("Masks (after merging) shape :", tf.shape(c_masks))
        
        ctx_types = tf.tile(ctx_types, [q_batch_size]) + n_doc_per_batch * ctx_add_type
    
    types   = tf.concat([tf.fill([tf.shape(question)[1]], 0), ctx_types], axis = -1)
    
    memory  = tf.concat([question, contexts], axis = 1)
    masks   = tf.concat([q_mask, c_masks], axis = -1) if q_mask is not None else None
    types   = tf.tile(tf.expand_dims(types, axis = 0), [q_batch_size, 1])

    return (memory, masks, types)

class MAGEncoder(BartEncoder):
    default_params  = HParamsMAGEncoder
    _attr_to_set    = TextTransformerEncoder._attr_to_set + [
        'subsample_at', 'subsample_after', 'subsampling_mode', 'subsampling_step',
        'subsampling_offset', 'max_types', 'random_training_type', 'repeat_pos_idx'
    ]
    
    def __init__(self, vocab_size, embedding_dim, name = None, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, name = name, ** kwargs
        )
        
        layer_idx = self.subsample_at
        if layer_idx < 0: layer_idx = len(self._layers) + layer_idx
        if self.subsample_after: layer_idx += 1
        self.M = max(0, min(len(self._layers), layer_idx))
        
        self.subsampling_layer  = None
        self.subsampling_drop_layer = tf.keras.layers.Dropout(
            self.hparams.subsampling_drop_rate
        ) if self.hparams.subsampling_drop_rate > 0 else None
        self.type_embedding_layer = None
        
        if self.subsampling_step > 1:
            if self.subsampling_mode not in _supported_subsamplings:
                raise ValueError("Unknown subsampling mode :\n  Got : {}\n  Accepted : {}".format(
                    self.subsampling_mode, _supported_subsamplings
                ))
            
            if self.subsampling_mode == 'conv':
                self.subsampling_layer = tf.keras.layers.Conv1D(
                    filters = self.embedding_dim, kernel_size = self.subsampling_step,
                    strides = self.subsampling_step, padding = 'valid', name = 'subsampling_layer'
                )
            elif self.subsampling_mode == 'separable':
                self.subsampling_layer = tf.keras.layers.SeparableConv1D(
                    filters = self.embedding_dim, kernel_size = self.subsampling_step,
                    strides = self.subsampling_step, padding = 'valid', name = 'subsampling_layer'
                )
            elif self.subsampling_mode == 'dense':
                self.subsampling_layer = tf.keras.layers.Dense(
                    units = self.embedding_dim, name = 'subsampling_layer'
                )
        
        if self.hparams.use_type_embedding:
            self.type_embedding_layer = FasterEmbedding(
                self.max_types, self.embedding_dim, name = "type_embedding"
            )
    
    def _maybe_init_subsampling_layer(self):
        if self.subsampling_layer is None or self.subsampling_mode != 'dense': return
        
        w = np.zeros(self.subsampling_layer.weights[0].shape)
        for i in range(self.embedding_dim):
            w[i::self.embedding_dim, i] = 1
        w /= self.subsampling_step
        self.subsampling_layer.set_weights([w, np.zeros(self.subsampling_layer.weights[1].shape)])

    def _build(self):
        super()._build()
        self._maybe_init_subsampling_layer()

    @property
    def embedding_layers(self):
        return self._layers[: self.M]
    
    @property
    def memory_layers(self):
        return self._layers[self.M :]
    
    @property
    def dummy_inputs(self):
        batch_size, q_seq_len, c_seq_len = 2, 16, 32
        
        q_tokens    = tf.ones([batch_size, q_seq_len], dtype = tf.int32)
        q_length    = tf.fill([batch_size, 1], q_seq_len)
        
        c_tokens    = tf.ones([batch_size, c_seq_len], dtype = tf.int32)
        c_length    = tf.fill([batch_size, 1], c_seq_len)
        
        return [q_tokens, q_length, c_tokens, c_length]
    
    @timer
    def pad_to_multiple(self, output, mask = None):
        output = pad_to_multiple(output, self.subsampling_step, axis = 1)
        if mask is not None:
            mask = pad_to_multiple(mask, self.subsampling_step, axis = -1)
        
        return output, mask

    @timer
    def subsample(self, output, mask = None, training = False):
        if self.subsampling_step <= 1: return output, mask
        
        if self.subsampling_drop_layer is not None:
            output = self.subsampling_drop_layer(output, training = training)
        
        if self.subsampling_mode == 'select':
            indices = tf.range(self.subsampling_offset, tf.shape(output)[1], self.subsampling_step)
            indices = tf.tile(tf.expand_dims(indices, axis = 0), [tf.shape(output)[0], 1])

            output = tf.gather(output, indices, batch_dims = 1)

            if mask is not None:
                mask = tf.gather(tf.squeeze(mask, [1, 2]), indices, batch_dims = 1)
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1])
        elif self.subsampling_mode in ('conv', 'separabl'):
            output = self.subsampling_layer(output, training = training)
            

            if mask is not None:
                indices = tf.range(0, tf.shape(output)[1]) * self.subsampling_step
                indices = tf.tile(tf.expand_dims(indices, axis = 0), [tf.shape(output)[0], 1])

                mask = tf.gather(tf.squeeze(mask, [1, 2]), indices, batch_dims = 1)
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1])
        elif self.subsampling_mode == 'dense':
            output, mask = self.pad_to_multiple(output, mask)
            
            output = tf.reshape(
                output, [tf.shape(output)[0], -1, self.subsampling_step * tf.shape(output)[-1]]
            )
            output = self.subsampling_layer(output)
            
            if mask is not None:
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1, self.subsampling_step])
                mask = tf.reduce_min(mask, axis = -1)
        else:
            output, mask = self.pad_to_multiple(output, mask)
            
            output = tf.reshape(
                output, [tf.shape(output)[0], -1, self.subsampling_step, tf.shape(output)[-1]]
            )
            
            if mask is not None:
                mask = tf.reshape(mask, [tf.shape(output)[0], 1, 1, -1, self.subsampling_step])
                mask = tf.reduce_min(mask, axis = -1)
            
            if self.subsampling_mode == 'min':
                output = tf.reduce_min(output, axis = 2)
            elif self.subsampling_mode == 'max':
                output = tf.reduce_max(output, axis = 2)
            else:
                output = tf.reduce_mean(output, axis = 2)
        
        return output, mask

    @timer
    def embed_types(self, memory, types, training = False, debug = False, ** kwargs):
        if self.type_embedding_layer is None: return memory, types
        
        if self.max_types == 2:
            types = tf.cast(types > 0, tf.int32)
        elif self.random_training_type and training and tf.reduce_max(types) < self.max_types:
            random_offset = tf.random.uniform(
                (tf.shape(types)[0], 1),
                minval  = 0,
                maxval  = self.max_types - tf.reduce_max(types),
                dtype   = tf.int32
            )
            types = types + (random_offset * tf.cast(types > 0, tf.int32))
        
        if debug:
            tf.print("Types used :", types)
        
        memory = memory + self.type_embedding_layer(types)
        
        return memory, types
    
    @timer
    def embed(self,
              inputs,
              input_length  = None,
              token_types   = None,
              position_ids  = None,
              
              mask  = None,
              training  = False,
              padding_mask  = None,
              look_ahead_mask   = None,
              
              positional_offset = -1,
              force_not_subsampling = False,
              
              return_state       = None,
              return_attention   = None,
              return_hidden_states   = None,
              return_mask        = None,
              as_dict    = False,
              
              debug = False,
              ** kwargs
             ):
        if return_state is None:            return_state = self.return_state
        if return_attention is None:        return_attention = self.return_attention
        if return_hidden_states is None:    return_hidden_states = self.return_hidden_states
        if return_mask is None:             return_mask = self.return_mask
        
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) % 2 == 0
            
            if len(inputs) > 2:
                if not isinstance(force_not_subsampling, (list, tuple)):
                    force_not_subsampling = [force_not_subsampling] * (len(inputs) // 2)
                if not isinstance(positional_offset, (list, tuple)):
                    positional_offset = [positional_offset] * (len(inputs) // 2)
                
                assert len(force_not_subsampling) == len(inputs) // 2, '{} vs {}'.format(len(force_not_subsampling), len(inputs))
                assert len(positional_offset) == len(inputs) // 2, '{} vs {}'.format(len(positional_offset), len(inputs))
                
                embeddings      = []
                states          = () if return_state else None
                attn_weights    = () if return_attention else None
                hidden_states   = () if return_hidden_states else None
                masks           = () if return_mask else None
                
                for i in range(0, len(inputs), 2):
                    outputs_i = self.embed(
                        inputs[i],
                        input_length    = inputs[i+1],
                        token_types     = token_types[i // 2] if token_types is not None else None,
                        position_ids    = position_ids[i // 2] if position_ids is not None else None,
                        training    = training,
                        positional_offset   = positional_offset[i // 2],
                        force_not_subsampling   = force_not_subsampling[i // 2],
                        
                        return_state    = return_state,
                        return_attention    = return_attention,
                        return_hidden_states    = return_hidden_states,
                        return_mask = return_mask,
                        as_dict = True,
                        
                        debug   = debug,
                        ** kwargs
                    )
                    embeddings.append(outputs_i.output)
                    if return_state:
                        states = states + (outputs_i.state, )
                    if return_attention:
                        attn_weights = attn_weights + (outputs_i.attention_weights, )
                    if return_hidden_states:
                        hidden_states = hidden_states + (outputs_i.hidden_states, )
                    if return_mask:
                        masks = masks + (outputs_i.mask, )
                
                return format_output(
                    output  = embeddings,
                    state   = states,
                    attn_weights    = attn_weights,
                    hidden_states   = hidden_states,
                    mask        = masks,
                    
                    return_state    = return_state,
                    return_attention    = return_attention,
                    return_hidden_states    = return_hidden_states,
                    return_mask = return_mask,
                    as_dict = as_dict
                )
            
            text, input_length = inputs
        else:
            text = inputs

        if debug:
            tf.print("Input tokens shape :", tf.shape(text), "-", input_length)
        
        batch_size = tf.shape(text)[0]
        n_doc_per_batch = -1
        if len(tf.shape(text)) == 3:
            n_doc_per_batch = tf.shape(text)[1]
            text            = tf.reshape(text, [-1, tf.shape(text)[-1]])
            input_length    = tf.reshape(input_length, [-1])
            if debug:
                tf.print("Input tokens reshaped shape :", tf.shape(text))
        
        states              = () if return_state else None
        attention_weights   = {} if return_attention else None
        hidden_states       = {} if return_hidden_states else None
        
        if mask is None:
            mask = build_mask(
                text, self.use_causal_attention, input_length = input_length,
                look_ahead_mask = look_ahead_mask, padding_mask = padding_mask
            )
        
        embedded = self.embeddings(
            text,
            input_length    = input_length,
            
            repeat_position = self.repeat_pos_idx,
            positional_offset   = positional_offset,
            
            training    = training,
            mask    = mask,
            debug   = debug,
            ** kwargs
        )
        
        output = embedded
        for i, layer in enumerate(self.embedding_layers):
            output, state, attn_weights = layer(
                output,
                input_length    = input_length,
                training    = training,
                mask    = mask,
                return_attention    = True,
                return_state        = True
            )
            if return_state:
                states  = states + (state, )
            
            if return_attention:
                if not isinstance(attn_weights, tuple):
                    attention_weights['attn_{}'.format(layer.name)] = attn_weights
                else:
                    attention_weights['attn_{}'.format(layer.name)] = attn_weights[0]
                    attention_weights['enc_attn_{}'.format(layer.name)] = attn_weights[1]
            
            if return_hidden_states:
                hidden_states['state_{}'.format(layer.name)] = output
        
        if not force_not_subsampling:
            output, mask = self.subsample(output, mask = mask, training = training)

            if debug:
                tf.print("Output subsampled shape :", tf.shape(output))
        
        if n_doc_per_batch != -1:
            output  = tf.reshape(
                output, [batch_size, n_doc_per_batch, tf.shape(output)[1], tf.shape(output)[-1]]
            )
            mask    = tf.reshape(mask,   [batch_size, n_doc_per_batch, 1, 1, tf.shape(mask)[-1]])

            if debug:
                tf.print("Output reshaped shape :", tf.shape(output))
        
        return format_output(
            output  = output,
            state   = states,
            attn_weights    = attention_weights,
            hidden_states   = hidden_states,
            mask        = mask,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = as_dict
        )
    
    @timer
    def process_memory(self,
                       embeddings,
                       mask = None,
                       training = False,
                       
                       return_state       = None,
                       return_attention   = None,
                       return_hidden_states   = None,
                       return_mask        = None,
                       as_dict    = False,
              
                       ** kwargs
                      ):
        if return_state is None:            return_state = self.return_state
        if return_attention is None:        return_attention = self.return_attention
        if return_hidden_states is None:    return_hidden_states = self.return_hidden_states
        if return_mask is None:             return_mask = self.return_mask
        
        states              = () if return_state else None
        attention_weights   = {} if return_attention else None
        hidden_states       = {} if return_hidden_states else None

        memory, mask, types = concat_qc(embeddings, mask = mask, training = training, ** kwargs)
        
        memory, types = self.embed_types(memory, types, training = training, ** kwargs)
        
        output = memory
        for i, layer in enumerate(self.memory_layers):
            output, state, attn_weights = layer(
                output,
                training    = training,
                padding_mask    = mask,
                return_attention    = True,
                return_state        = True
            )
            if return_state:
                states  = states + (state, )
            
            if return_attention:
                if not isinstance(attn_weights, tuple):
                    attention_weights['memory_attn_{}'.format(layer.name)] = attn_weights
                else:
                    attention_weights['memory_attn_{}'.format(layer.name)] = attn_weights[0]
                    attention_weights['memory_enc_attn_{}'.format(layer.name)] = attn_weights[1]
            
            if return_hidden_states:
                hidden_states['state_{}'.format(layer.name)] = output
        
        return format_output(
            output  = output,
            state   = states,
            attn_weights    = attention_weights,
            hidden_states   = hidden_states,
            mask        = mask,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = as_dict
        )
    
    @timer
    def call(self,
             inputs,
             mask       = None,
             training   = False,
             
             merge_contexts     = False,
             positional_offset  = -1, 
             
             return_state       = None,
             return_attention   = None,
             return_hidden_states   = None,
             return_mask        = None,
             as_dict    = False,
             
             ** kwargs
            ):
        if return_state is None:            return_state = self.return_state
        if return_attention is None:        return_attention = self.return_attention
        if return_hidden_states is None:    return_hidden_states = self.return_hidden_states
        if return_mask is None:             return_mask = self.return_mask

        memory_outputs = self.embed(
            inputs,
            mask    = mask,
            training    = training,
            positional_offset   = positional_offset,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = True,
            as_dict = True,
            ** kwargs
        )
        embeddings, masks = memory_outputs.output, memory_outputs.mask
        
        outputs = self.process_memory(
            embeddings,
            mask    = masks,
            training    = training,
            merge_contexts  = merge_contexts,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = True,
            ** kwargs
        )
        
        return format_output(
            outputs.output,
            state   = (memory_outputs.state, outputs.state),
            attn_weights    = (memory_outputs.attention_weights, outputs.attention_weights),
            hidden_states   = (memory_outputs.hidden_states, outputs.hidden_states),
            mask    = outputs.mask,
            
            return_state    = return_state,
            return_attention    = return_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = as_dict
        )
    
class MAG(Bart):
    encoder_class   = MAGEncoder
    default_params  = HParamsMAG
    
    @property
    def dummy_inputs(self):
        batch_size, q_in_seq_len, c_in_seq_len, out_seq_len = 2, 16, 32, 8
        
        q_tokens    = tf.ones([batch_size, q_in_seq_len], dtype = tf.int32)
        q_length    = tf.fill([batch_size, 1], q_in_seq_len)
        
        c_tokens    = tf.ones([batch_size, c_in_seq_len], dtype = tf.int32)
        c_length    = tf.fill([batch_size, 1], c_in_seq_len)
        
        text = tf.ones([batch_size, out_seq_len], dtype = tf.int32)
        text_length = tf.fill([batch_size, 1], out_seq_len)
        
        return [q_tokens, q_length, c_tokens, c_length, text, text_length]
    
    def _build(self):
        super()._build()
        self.encoder._maybe_init_subsampling_layer()

custom_objects  = {
    'MAG'   : MAG
}

custom_functions    = custom_objects
