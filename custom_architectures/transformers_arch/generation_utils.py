
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
import collections
import tensorflow as tf

from loggers import timer
from utils import get_object
from custom_layers import log_softmax
from utils.text import create_padding_mask
from custom_architectures.transformers_arch.transformer_arch import format_output

TransformerInferenceOutput = collections.namedtuple(
    "TransformerInferenceOutput", [
        "tokens",
        "output",
        "score"
    ]
)

TransformerInferenceState   = collections.namedtuple(
    "TransformerInferenceState", [
        "tokens",
        "input_length",
        "padding_mask",
        "finished",
        "logits",
        "scores"
    ]
)
def get_shape_invariant():
    return TransformerInferenceState(
        tokens          = tf.TensorSpec(shape = (None, None),       dtype = tf.int32),
        input_length    = tf.TensorSpec(shape = (None, 1),          dtype = tf.int32),
        padding_mask    = tf.TensorSpec(shape = (None, 1, 1, None), dtype = tf.float32),
        finished        = tf.TensorSpec(shape = (None, ),           dtype = tf.int32),
        logits          = tf.TensorSpec(shape = (None, None, None), dtype = tf.float32),
        scores          = tf.TensorSpec(shape = (None, ),           dtype = tf.float32)
    )

@timer
def infer(model,
          * args,
          method = 'greedy',
          
          max_length    = None,

          return_state       = None,
          return_attention   = None,
          return_hidden_states   = None,
          return_mask        = None,
          as_dict    = False,

          ** kwargs
         ):
    if max_length is None:              max_length = model.max_input_length
    if return_state is None:            return_state = model.return_state
    if return_attention is None:        return_attention = model.return_attention
    if return_hidden_states is None:    return_hidden_states = model.return_hidden_states
    if return_mask is None:             return_mask = model.return_mask

    return get_object(
        _inference_methods,
        method,
        * args,
            
        model = model,
        max_length      = max_length,
        return_state    = return_state,
        return_attention    = return_attention,
        return_hidden_states    = return_hidden_states,
        return_mask         = return_mask,
        as_dict     = as_dict,
        
        ** kwargs
    )

def _infer(model,
           tokens    = None,
           input_length  = None,
           encoder_output    = None,
           initial_state     = None,
          
           enc_padding_mask  = None,
           padding_mask  = None,
           training  = False,
           use_cache = False,
           
           sos_token    = None,
           eos_token    = None,
           max_length   = None,
           use_sampling = False,
           early_stopping    = True,

           return_state       = None,
           return_attention   = None,
           return_hidden_states   = None,
           return_mask        = None,
           as_dict    = False,

           ** kwargs
          ):
    def cond(tokens, input_length, padding_mask, finished, logits, scores):
        return not (early_stopping and tf.reduce_sum(finished) == batch_size)
    
    def body(tokens, input_length, padding_mask, finished, logits, scores):
        outputs = model(
            tokens,
            input_length    = input_length,
            encoder_output  = encoder_output,
            padding_mask    = padding_mask,
            enc_padding_mask    = enc_padding_mask,
            positional_offset   = -1 if not use_cache else input_length - 1 + model.positional_offset,
            training    = training,
            
            return_state    = use_cache,
            return_attention    = return_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = True,
            
            ** kwargs
        )
        logits      = outputs.output
        
        next_token  = _select_next_token(
            logits[:,-1,:], n = 1, previous = tokens, use_sampling = use_sampling
        )
        next_token  = tf.reshape(
            tf.cast(next_token, tokens.dtype), [batch_size, 1]
        )

        tokens  = tf.concat([tokens, next_token], axis = -1)

        finished    = tf.maximum(
            finished, tf.cast(tf.math.equal(next_token[:,0], eos_token), tf.int32)
        )
        
        input_length += 1 - tf.expand_dims(finished, axis = 1)
        padding_mask = tf.concat([
            padding_mask,
            tf.reshape(tf.cast(finished, padding_mask.dtype), [-1, 1, 1, 1])
        ], axis = -1)
        
        return TransformerInferenceState(
            tokens  = tokens,
            input_length    = input_length,
            padding_mask    = padding_mask,
            finished    = finished,
            logits      = logits,
            scores      = scores
        )
    
    batch_size  = 1
    if encoder_output is not None:
        batch_size = tf.shape(encoder_output)[0]
    
    if tokens is None:
        tokens          = tf.fill((batch_size, 1), sos_token)
        input_length    = tf.fill((batch_size, 1), 1)
    elif isinstance(tokens, (list, tuple)):
        tokens, input_length    = tokens
    
    if batch_size == 1: batch_size = tf.shape(tokens)[0]
    
    if input_length is None:
        input_length    = tf.fill((batch_size,), tf.shape(tokens)[1])
    
    if padding_mask is None:
        padding_mask    = create_padding_mask(tokens, seq_len = input_length, dtype = tf.float32)

    outputs = tf.while_loop(
        cond    = cond,
        body    = body,
        loop_vars   = TransformerInferenceState(
            tokens  = tokens,
            input_length    = input_length,
            padding_mask    = padding_mask,
            finished    = tf.zeros((batch_size,), dtype = tf.int32),
            logits      = tf.zeros((batch_size, 1, tf.shape(encoder_output)[-1])),
            scores      = tf.zeros((batch_size, ))
        ),
        shape_invariants    = get_shape_invariant(),
        maximum_iterations  = max_length
    )
    
    return TransformerInferenceOutput(
        tokens  = outputs.tokens[..., 1:],
        score   = _score_output(outputs.logits, outputs.tokens[..., 1:]),
        output  = outputs.logits
    )

def _infer_beam_search(model,
                       vocab_size,
                       tokens    = None,
                       input_length  = None,
                       encoder_output    = None,
                       initial_state     = None,

                       temperature  = 1.,
                       length_power = 0.,
                       num_beams    = 10,
                       num_sentences    = 5,

                       enc_padding_mask  = None,
                       padding_mask  = None,
                       training  = False,
                       use_cache = False,

                       sos_token    = None,
                       eos_token    = None,
                       max_length   = None,
                       use_sampling = False,
                       early_stopping    = True,

                       return_state       = None,
                       return_attention   = None,
                       return_hidden_states   = None,
                       return_mask        = None,
                       as_dict    = False,

                       ** kwargs
                      ):
    def cond(tokens, input_length, padding_mask, finished, logits, scores):
        if not early_stopping: return True

        finished_per_batch = tf.reshape(finished, [batch_size, -1])
        return tf.reduce_sum(finished_per_batch[:,:num_sentences]) != batch_size * num_sentences
    
    def body(tokens, input_length, padding_mask, finished, logits, scores):
        outputs = model(
            tokens,
            input_length    = input_length,
            encoder_output  = encoder_output,
            padding_mask    = padding_mask,
            enc_padding_mask    = enc_padding_mask,
            positional_offset   = -1 if not use_cache else input_length - 1 + model.positional_offset,
            training    = training,
            apply_softmax   = False,
            
            return_state    = use_cache,
            return_attention    = return_attention,
            return_hidden_states    = return_hidden_states,
            return_mask = return_mask,
            as_dict = True,
            
            ** kwargs
        )
        finish_mask = tf.expand_dims(tf.cast(1 - finished, tf.float32), axis = 1)

        logits  = outputs.output[:, -1, :]
        if temperature != 1.:
            logits = logits / temperature
        
        beam_scores = log_softmax(logits, axis = -1) * finish_mask
        beam_scores = beam_scores + tf.expand_dims(scores, axis = 1)
        
        if tf.shape(tokens)[1] == 1:
            mask = tf.expand_dims(
                tf.cast(tf.tile(tf.range(num_beams), [batch_size]) == 0, tf.float32), axis = -1
            )
        else:
            mask    = tf.cast(tf.range(vocab_size) != eos_token, tf.float32)
            mask    = tf.tile(tf.expand_dims(mask, 0), [effective_batch_size, 1])
            mask    = 1. - (mask * tf.expand_dims(tf.cast(finished, tf.float32), axis = 1))

        reshaped_scores = beam_scores * mask + (1. - mask) * -1e4
        reshaped_scores = reshaped_scores / tf.cast(input_length, tf.float32) ** length_power
        reshaped_scores = tf.reshape(reshaped_scores, [batch_size, -1])
        
        next_token  = _select_next_token(
            reshaped_scores, n = num_beams, previous = None, use_sampling = use_sampling
        )
        next_token  = tf.reshape(
            tf.cast(next_token, tokens.dtype), [effective_batch_size, 1]
        )
        token_batch_idx = tf.reshape(next_token // vocab_size, [-1]) + batch_idx_add
        next_token      = next_token % vocab_size
        
        beam_scores     = tf.gather(beam_scores,    token_batch_idx)
        input_length    = tf.gather(input_length,   token_batch_idx)
        finished        = tf.gather(finished,       token_batch_idx)
        padding_mask    = tf.gather(padding_mask,   token_batch_idx)
        
        scores          = tf.reshape(
            tf.gather(beam_scores, next_token, batch_dims = 1, axis = -1), [-1]
        )
        tokens  = tf.concat([
            tf.gather(tokens, token_batch_idx),
            next_token
        ], axis = -1)

        finished    = tf.maximum(
            finished, tf.cast(tf.math.equal(next_token[:,0], eos_token), tf.int32)
        )
        
        input_length += 1 - tf.expand_dims(finished, axis = 1)
        padding_mask = tf.concat([
            padding_mask,
            tf.reshape(tf.cast(finished, padding_mask.dtype), [-1, 1, 1, 1])
        ], axis = -1)
        
        return TransformerInferenceState(
            tokens  = tokens,
            input_length    = input_length,
            padding_mask    = padding_mask,
            finished    = finished,
            logits      = outputs.output,
            scores      = scores
        )

    batch_size  = 1
    if encoder_output is not None:
        batch_size = tf.shape(encoder_output)[0]

        encoder_output  = tf.repeat(encoder_output, num_beams, axis = 0)
        if enc_padding_mask is not None:
            enc_padding_mask  = tf.repeat(enc_padding_mask, num_beams, axis = 0)
    
    effective_batch_size    = batch_size * num_beams

    if tokens is None:
        tokens          = tf.fill((effective_batch_size, 1), sos_token)
        input_length    = tf.fill((effective_batch_size, 1), 1)
    
    if input_length is None:
        input_length    = tf.fill((effective_batch_size,), tf.shape(tokens)[1])
    
    if padding_mask is None:
        padding_mask    = create_padding_mask(tokens, seq_len = input_length, dtype = tf.float32)
    
    batch_idx_add   = tf.repeat(tf.range(batch_size), num_beams, axis = 0) * num_beams
    
    outputs = tf.while_loop(
        cond    = cond,
        body    = body,
        loop_vars   = TransformerInferenceState(
            tokens  = tokens,
            input_length    = input_length,
            padding_mask    = padding_mask,
            finished    = tf.zeros((effective_batch_size,), dtype = tf.int32),
            logits      = tf.zeros((effective_batch_size, 1, tf.shape(encoder_output)[-1])),
            scores      = tf.zeros((effective_batch_size, ))
        ),
        shape_invariants    = get_shape_invariant(),
        maximum_iterations  = max_length
    )
    
    return TransformerInferenceOutput(
        tokens  = tf.reshape(outputs.tokens, [batch_size, num_beams, -1])[:, :num_sentences, 1:],
        score   = tf.reshape(outputs.scores, [batch_size, num_beams])[:, :num_sentences],
        output  = tf.reshape(outputs.logits, [batch_size, num_beams, tf.shape(outputs.logits)[1], -1])[:, :num_sentences],
    )

def _score_output(probs, indices):
    return tf.reduce_mean(
        tf.math.log(tf.gather(probs, indices, batch_dims = 2, axis = -1)), axis = -1
    )

def _select_next_token(scores, n = 1, previous = None, use_sampling = False):
    if not use_sampling:
        if n == 1: return tf.argmax(scores, axis = -1)
        return tf.nn.top_k(scores, k = n).indices
    
    raise NotImplementedError()
    
_inference_methods  = {
    'greedy'    : _infer,
    'sample'    : lambda * args, ** kwargs: _infer(* args, use_sampling = True, ** kwargs),
    'beam'      : _infer_beam_search
}