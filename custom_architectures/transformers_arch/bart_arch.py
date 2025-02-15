
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

""" TF 2.0 BART model, compatible with the `transformers`' model. """

import numpy as np
import tensorflow as tf

from loggers import timer
from custom_layers import FasterEmbedding, get_activation
from custom_architectures.transformers_arch.embedding_head import EmbeddingHead, HParamsEmbeddingHead
from custom_architectures.transformers_arch.text_transformer_arch import *

_supported_subsamplings = ('select', 'conv', 'separable', 'dense', 'min', 'max', 'mean')

HParamsBartEncoder      = HParamsTextTransformerEncoder
HParamsBartEmbedding    = HParamsBartEncoder(** HParamsEmbeddingHead)

HParamsBartDecoder  = HParamsTextTransformerDecoder(
    final_activation    = 'softmax',
)

HParamsBart     = HParamsTextTransformer(
    ** HParamsBartEncoder.get_config(add_prefix = 'encoder'),
    ** HParamsBartDecoder.get_config(add_prefix = 'decoder')
)

class BartEncoder(TextTransformerEncoder):
    default_params = HParamsBartEncoder
    
    def transfer_weights(self, pretrained):
        from models.weights_converter import partial_transfer_learning
        offset, n_enc_layer_weights = 1, 16
        
        weights = pretrained.model.encoder.get_weights()
        # Invert `key` and `value` weights for each MHA layer
        for i in range(pretrained.config.encoder_layers):
            weights[i * n_enc_layer_weights + offset], weights[i * n_enc_layer_weights + offset + 2] = (
                weights[i * n_enc_layer_weights + offset + 2], weights[i * n_enc_layer_weights + offset]
            )
            weights[i * n_enc_layer_weights + offset + 1], weights[i * n_enc_layer_weights + offset + 3] = (
                weights[i * n_enc_layer_weights + offset + 3], weights[i * n_enc_layer_weights + offset + 1]
            )
        weights = [pretrained.get_weights()[0], weights[0]] + weights[-2:] + weights[1:-2]
        
        partial_transfer_learning(self, weights)

    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'facebook/bart-large',
                        pretrained_task = 'generation',
                        pretrained      = None,
                        ** kwargs
                       ):
        if pretrained is None:
            with tf.device('cpu') as d:
                pretrained = transformers_bart(pretrained_name, pretrained_task)

        config = HParamsBartEncoder(
            vocab_size      = pretrained.config.vocab_size,
            embedding_dim   = pretrained.config.d_model,
            max_input_length    = pretrained.config.max_position_embeddings,
            positional_offset   = 2,
            scale_embedding = False,
            epsilon = 1e-5,

            num_layers  = pretrained.config.encoder_layers,
            ffn_dim     = pretrained.config.encoder_ffn_dim,
            ffn_activation  = pretrained.config.activation_function,
            mha_num_heads   = pretrained.config.encoder_attention_heads
        )
        
        instance = cls(** config(** kwargs))
        instance._build()

        instance.transfer_weights(pretrained)
        
        return instance

class BartEmbedding(BartEncoder):
    default_params = HParamsBartEmbedding
    
    def __init__(self, output_dim, vocab_size, embedding_dim, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, ** kwargs
        )
        
        self.embedding_head = EmbeddingHead(** self.hparams)

    def compute_output(self, output, training = False, mask = None, ** kwargs):
        return self.embedding_head(output, mask = mask, training = training)

class BartDecoder(TextTransformerDecoder):
    default_params = HParamsBartDecoder
    
    def __init__(self, vocab_size, embedding_dim, token_embedding = None, name = None, ** kwargs):
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim,
            token_embedding = token_embedding, ** kwargs
        )

        self.final_bias = self.add_weight(
            shape = [1, vocab_size], name = "final_bias", trainable = False, initializer = "zeros"
        )
        self.final_act_layer    = get_activation(self.hparams.final_activation)
    
    @timer
    def compute_output(self, output, training = False, mask = None, apply_softmax = True,
                       ** kwargs):
        output = self.embeddings.linear(output) + self.final_bias
        if self.final_act_layer is not None and apply_softmax:
            output = self.final_act_layer(output)
        return output
        
    def transfer_weights(self, pretrained):
        from models.weights_converter import partial_transfer_learning, print_vars
        offset, n_enc_layer_weights = 2, 16
        
        weights = pretrained.model.decoder.get_weights()
        # Invert `key` and `value` weights for each MHA layer
        offset, n_mha_weights, n_dec_layer_weights = 1, 10, 26
        for i in range(pretrained.config.decoder_layers):
            weights[i * n_dec_layer_weights + offset], weights[i * n_dec_layer_weights + offset + 2] = (
                weights[i * n_dec_layer_weights + offset + 2], weights[i * n_dec_layer_weights + offset]
            )
            weights[i * n_dec_layer_weights + offset + 1], weights[i * n_dec_layer_weights + offset + 3] = (
                weights[i * n_dec_layer_weights + offset + 3], weights[i * n_dec_layer_weights + offset + 1]
            )
            
            weights[i * n_dec_layer_weights + n_mha_weights + offset], weights[i * n_dec_layer_weights + n_mha_weights + offset + 2] = (
                weights[i * n_dec_layer_weights + n_mha_weights + offset + 2],
                weights[i * n_dec_layer_weights + n_mha_weights + offset]
            )
            weights[i * n_dec_layer_weights + n_mha_weights + offset + 1], weights[i * n_dec_layer_weights + n_mha_weights + offset + 3] = (
                weights[i * n_dec_layer_weights + n_mha_weights + offset + 3],
                weights[i * n_dec_layer_weights + n_mha_weights + offset + 1]
            )
        # Add shared embeddings weights to the list
        weights = [pretrained.get_weights()[0], weights[0]] + weights[-2:] + weights[1:-2] + [pretrained.get_weights()[-1]]
        
        partial_transfer_learning(self, weights)

    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'facebook/bart-large',
                        pretrained_task = 'generation',
                        pretrained      = None,
                        ** kwargs
                       ):
        if pretrained is None:
            with tf.device('cpu') as d:
                pretrained = transformers_bart(pretrained_name, pretrained_task)

        config = HParamsBartDecoder(
            vocab_size      = pretrained.config.vocab_size,
            embedding_dim   = pretrained.config.d_model,
            max_input_length    = pretrained.config.max_position_embeddings,
            positional_offset   = 2,
            scale_embedding = False,
            epsilon     = 1e-5,
            sos_token   = 0,
            eos_token   = 2,

            num_layers  = pretrained.config.decoder_layers,
            ffn_dim     = pretrained.config.decoder_ffn_dim,
            ffn_activation  = pretrained.config.activation_function,
            mha_num_heads   = pretrained.config.decoder_attention_heads,
            enc_mha_num_heads   = pretrained.config.decoder_attention_heads
        )
        
        instance = cls(** config(** kwargs))
        instance._build()
        
        instance.transfer_weights(pretrained)
        
        return instance

class Bart(TextTransformer):
    encoder_class   = BartEncoder
    decoder_class   = BartDecoder
    default_params  = HParamsBart
    
    def __init__(self, vocab_size, embedding_dim, max_input_length,
                 ** kwargs):
        shared_embedding = FasterEmbedding(vocab_size, embedding_dim, name = "token_embedding")
        super().__init__(
            vocab_size = vocab_size, embedding_dim = embedding_dim, max_input_length = max_input_length,
            shared_layers = {'token_embedding' : shared_embedding}, ** kwargs
        )

    @classmethod
    def from_pretrained(cls,
                        pretrained_name = 'facebook/bart-large',
                        pretrained_task = 'generation', 
                        pretrained      = None,
                        ** kwargs
                       ):
        if pretrained is None:
            with tf.device('cpu') as d:
                pretrained = transformers_bart(pretrained_name, pretrained_task)

        config = HParamsBart(
            vocab_size      = pretrained.config.vocab_size,
            embedding_dim   = pretrained.config.d_model,
            max_input_length    = pretrained.config.max_position_embeddings,
            positional_offset   = 2,
            scale_embedding = False,
            epsilon     = 1e-5,
            sos_token   = 0,
            eos_token   = 2,

            encoder_num_layers  = pretrained.config.encoder_layers,
            encoder_ffn_dim     = pretrained.config.encoder_ffn_dim,
            encoder_ffn_activation  = pretrained.config.activation_function,
            encoder_mha_num_heads   = pretrained.config.encoder_attention_heads,

            decoder_num_layers  = pretrained.config.decoder_layers,
            decoder_ffn_dim     = pretrained.config.decoder_ffn_dim,
            decoder_ffn_activation  = pretrained.config.activation_function,
            decoder_mha_num_heads   = pretrained.config.decoder_attention_heads,
            decoder_enc_mha_num_heads   = pretrained.config.decoder_attention_heads
        )
        
        instance = cls(** config(** kwargs))
        instance._build()
        
        instance.encoder.transfer_weights(pretrained)
        instance.decoder.transfer_weights(pretrained)
        
        return instance

def transformers_bart(name = 'facebook/bart-large', task = 'generation'):
    import transformers
    if task == 'generation':
        return transformers.TFBartForConditionalGeneration.from_pretrained(name)
    else:
        raise ValueError("Unknown task !\n  Accepted : {}\n  Got : {}".format(
            tuple(_transformers_pretrained_task.keys()), task
        ))

_bart_classes   = {
    'BartEncoder'   : BartEncoder,
    'BartEmbedding' : BartEmbedding,
    'BartDecoder'   : BartDecoder,
    'Bart'          : Bart
}
        
custom_functions    = {
    ** _bart_classes,
    'transformers_bart' : transformers_bart
}

custom_objects  = {
    ** _bart_classes,
    'TransformerEncoder'    : TransformerEncoder
}

_encoders   = {'Bart' : BartEmbedding}
_decoders   = {'Bart' : BartDecoder}
